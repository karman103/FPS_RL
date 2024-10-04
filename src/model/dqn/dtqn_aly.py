import torch
from .base import DQNModuleBase
from .DTQN.transformer import TransformerIdentityLayer, TransformerLayer
from .DTQN.position_encodings import PositionEncoding, PosEnum
from .DTQN.gates import GRUGate, ResGate
from .DTQN.representations import (
    ObservationEmbeddingRepresentation,
    ActionEmbeddingRepresentation,
)
from torch import nn

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
        module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.in_proj_bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class DTQN_aly(DQNModuleBase):
    """Deep Transformer Q-Network for partially observable reinforcement learning.
    ASSUMPTION 1: WE ARE NOT TAKING ACTION EMBEDDINGS FOR THE INPUT FOR TRANSFORMERS
    ASSUMPTION 2: WE ARE USING CNN OUTPUTS AS EMBEDDINGS, WHICH WILL HAVE obs_dim dimensions as inputs, and d_k as outputs
    Args:
        obs_dim:            The length of the observation vector.
        num_actions:        The number of possible environments actions.
        (NOT USED) embed_per_obs_dim:  Used for discrete observation space. Length of the embed for each
            element in the observation dimension.
        (NOT USED) action_dim:         The number of features to give the action.
        d_k:   The dimensionality of the network. Referred to as d_k by the
            original transformer.
        num_heads:          The number of heads to use in the MultiHeadAttention.
        num_layers:         The number of transformer blocks to use.
        history_len:        The maximum number of observations to take in.
        dropout:            Dropout percentage. Default: `0.0`
        gate:               Which layer to use after the attention and feedforward submodules (choices: `res`
            or `gru`). Default: `res`
        identity:           Whether or not to use identity map reordering. Default: `False`
        pos:                The kind of position encodings to use. `0` uses no position encodings, `1` uses
            learned position encodings, and `sin` uses sinusoidal encodings. Default: `1`
        (NOT USED) discrete:           Whether or not the environment has discrete observations. Default: `False`
        (NOT USED) vocab_sizes:        If discrete env only. Represents the number of observations in the
            environment. If the environment has multiple obs dims with different number
            of observations in each dim, this can be supplied as a vector. Default: `None`
    """
    def __init__(self, params, ffn_output=128, obs_dim=100, d_k=4672, num_heads=8, num_layers=5, history_len=50, dropout=0.1, gate='res', identity=False, pos='sin'):
        # Investigate what are params
        super(DTQN_aly, self).__init__(params)
        self.params = params
        self.obs_dim = obs_dim
        self.hidden_dim = params.hidden_dim
        self.ffn_output = ffn_output
        self.d_k = d_k
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.history_len = history_len
        self.dropout = dropout
        self.gate = gate
        self.identity = identity
        self.pos = pos
        pos_function_map = {
            PosEnum.LEARNED: PositionEncoding.make_learned_position_encoding,
            PosEnum.SIN: PositionEncoding.make_sinusoidal_position_encoding,
            PosEnum.NONE: PositionEncoding.make_empty_position_encoding,
        }
        self.position_embedding = pos_function_map[PosEnum(pos)](
            context_len=history_len, embed_dim=d_k
        )
        if gate == "gru":
            attn_gate = GRUGate(embed_size=d_k)
            mlp_gate = GRUGate(embed_size=d_k)
        elif gate == "res":
            attn_gate = ResGate()
            mlp_gate = ResGate()
        else:
            raise ValueError("Gate must be one of `gru`, `res`")

        self.dropout = nn.Dropout(dropout)
        if identity:
            transformer_block = TransformerIdentityLayer
        else:
            transformer_block = TransformerLayer
        self.transformer_layers = nn.Sequential(
            *[
                transformer_block(
                    num_heads,
                    d_k,
                    history_len,
                    dropout,
                    attn_gate,
                    mlp_gate,
                )
                for _ in range(num_layers)
            ]
        )
        self.ffn = nn.Sequential(
                nn.Linear(d_k, d_k),
                nn.ReLU(),
                nn.Linear(d_k, self.ffn_output),
                nn.ReLU(),
                nn.Linear(self.ffn_output, self.hidden_dim)
            )
        self.apply(init_weights)
    def forward(self, x_screens, x_variables):
        batch_size = x_screens.size(0)
        seq_len = x_screens.size(1)
        obs_dim = x_screens.size()[2:] if len(x_screens.size()) > 3 else x_screens.size(2)
        assert (
            seq_len <= self.history_len
        ), "Cannot forward, history is longer than expected."
        assert x_screens.ndimension() == 5
        assert len(x_variables) == self.n_variables
        assert all(x.ndimension() == 2 and x.size(0) == batch_size and
                   x.size(1) == seq_len for x in x_variables)

        obss_embeddings, output_gf = self.base_forward(
            x_screens.view(batch_size * seq_len, *x_screens.size()[2:]),
            [v.contiguous().view(batch_size * seq_len) for v in x_variables]
        ) 
        # Maybe we should have an extra embedding for actions inside of the input?
        obss_embeddings = obss_embeddings.reshape(batch_size, seq_len, obss_embeddings.size(-1)) # Size: (batch_size, histroy_len, d_k)
        
        inputs = self.dropout(
                obss_embeddings + self.position_embedding()[:, :seq_len, :]
            )
        output = self.ffn(self.transformer_layers(inputs)) # Size: (batch_size, histroy_len, self.hidden_dim)
        output_sc = self.head_forward(output.view(-1, self.hidden_dim))
        output_sc = output_sc.view(batch_size, seq_len, output_sc.size(1)) # Size: (batch_size, histroy_len, num_actions = 7)
        print("OUTPUT_SC: ", output_sc.size()) # Size (batch_size, num_actions)
        return output_sc, output_gf # NOT SURE ABOUT THAT

# To do:
# - Game features vs variables / What other variables should we be feeding to the DTQN
# - Checkout dtqn_agent.py -> function: f_train
# - Are the input / output shapes correct?




# Questions and uncertainties:
# - Are the shapes correct?
# - Do we proceed the same way as in an LSTM? i.e, do we input and output hidden states? Or just a bunch of observations? I.e. images and states? (ISSUE)
# - 