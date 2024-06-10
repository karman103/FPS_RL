import torch
import torch.nn as nn
from .base import DQNModuleBase, DQN
from DTQN.representations import ObservationEmbeddingRepresentation, ActionEmbeddingRepresentation
from DTQN.position_encodings import PosEnum, PositionEncoding
from DTQN.gates import GRUGate, ResGate
from DTQN.transformer import TransformerLayer, TransformerIdentityLayer


class DQNModuleTransformer(DQNModuleBase):

    def __init__(self, params):
        super(DQNModuleTransformer, self).__init__(params)

        # Initialize transfrmer-specific components -> I'd like to use existing CNN from Arnold project instead of this

        self.obs_embedding = ObservationEmbeddingRepresentation.make_image_representation(
            obs_dim=(params.n_fm, params.height, params.width),
            outer_embed_size=params.hidden_dim
        )
        
        if params.action_dim > 0:
            self.action_embedding = ActionEmbeddingRepresentation(
                num_actions=params.n_actions, 
                action_dim=params.action_dim
            )
        else:
            self.action_embedding = None

        pos_function_map = {
            PosEnum.LEARNED: PositionEncoding.make_learned_position_encoding,
            PosEnum.SIN: PositionEncoding.make_sinusoidal_position_encoding,
            PosEnum.NONE: PositionEncoding.make_empty_position_encoding,
        }
       
        self.position_embedding = pos_function_map[PosEnum(params.pos)](
            context_len=params.hist_size, 
            embed_dim=params.hidden_dim
        )

        if params.gate == "gru":
            attn_gate = GRUGate(embed_size=params.hidden_dim)
            mlp_gate = GRUGate(embed_size=params.hidden_dim)
        elif params.gate == "res":
            attn_gate = ResGate()
            mlp_gate = ResGate()
        else:
            raise ValueError("Gate must be one of `gru`, `res`")
        

        transformer_block = TransformerIdentityLayer if params.identity else TransformerLayer

        self.transformer_layers = nn.Sequential(*[
            transformer_block(
                num_heads=params.num_heads,
                embed_size=params.hidden_dim,
                history_len=params.hist_size,
                dropout=params.dropout,
                attn_gate=attn_gate,
                mlp_gate=mlp_gate,
            ) for _ in range(params.num_layers)
        ])

        # Bag attention mechanism - not sure if this will be necessary in our model
        self.bag_size = params.bag_size
        self.bag_attn_weights = None
        if self.bag_size > 0:
            self.bag_attention = nn.MultiheadAttention(
                params.hidden_dim,
                params.num_heads,
                dropout=params.dropout,
                batch_first=True,
            )
            self.ffn = nn.Sequential(
                nn.Linear(params.hidden_dim * 2, params.hidden_dim),
                nn.ReLU(),
                nn.Linear(params.hidden_dim, params.n_actions),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(params.hidden_dim, params.hidden_dim),
                nn.ReLU(),
                nn.Linear(params.hidden_dim, params.n_actions),
            )

        self.history_len = params.hist_size

    def forward(self, x_screens, x_variables, prev_state=None):
        """
        Argument sizes:
            - x_screens of shape (batch_size, seq_len, n_fm, h, w)
            - x_variables list of n_var tensors of shape (batch_size, seq_len)
        """
        batch_size = x_screens.size(0)
        seq_len = x_screens.size(1)

        # Flattening seq_len into batch_size ensures that it will be applied to all timesteps independently.
        state_input, output_gf = self.base_forward(
            x_screens.view(batch_size * seq_len, *x_screens.size()[2:]),
            [v.contiguous().view(batch_size * seq_len) for v in x_variables]
        )

        # unflatten the input and apply the transformer
        transformer_input = state_input.view(batch_size, seq_len, self.output_dim)
        transformer_output = self.transformer_layers(
            transformer_input + self.position_embedding()[:, :seq_len, :]
        )

        # apply the head to transformer hidden states (simulating larger batch again)
        output_sc = self.head_forward(transformer_output.view(-1, self.hidden_dim))

        # unflatten scores and game features
        output_sc = output_sc.view(batch_size, seq_len, output_sc.size(1))
        if self.n_features:
            output_gf = output_gf.view(batch_size, seq_len, self.n_features)

        return output_sc, output_gf, None
    


class DQNTransformer(DQN):

    DQNModuleClass = DQNModuleTransformer

    def __init__(self, params):
        super(DQNTransformer, self).__init__(params)

    def reset(self):
        # no need to reset any hidden states for transformer
        pass

    def f_eval(self, last_states):
        screens, variables = self.prepare_f_eval_args(last_states)
        
        # feed the last `hist_size` ones
        output = self.module(
            screens.view(1, self.hist_size, *self.screen_shape),
            [variables[:, i].contiguous().view(1, self.hist_size)
             for i in range(self.params.n_variables)],
        )

        return output[:-1]
    
    def f_train(self, screens, variables, features, actions, rewards, isfinal, loss_history=None):
        screens, variables, features, actions, rewards, isfinal = self.prepare_f_train_args(screens, variables, features, actions, rewards, isfinal)

        batch_size = self.params.batch_size
        seq_len = self.hist_size + self.params.n_rec_updates

        output_sc, output_gf, _ = self.module(
            screens,
            [variables[:, :, i] for i in range(self.params.n_variables)],
        )

        # compute scores
        mask = torch.ByteTensor(output_sc.size()).fill_(0)
        for i in range(batch_size):
            for j in range(seq_len - 1):
                mask[i, j, int(actions[i, j])] = 1
        scores1 = output_sc.masked_select(self.get_var(mask))
        scores2 = rewards + (
            self.params.gamma * output_sc[:, 1:, :].max(2)[0] * (1 - isfinal)
        )

        # dqn loss
        loss_sc = self.loss_fn_sc(
            scores1.view(batch_size, -1)[:, -self.params.n_rec_updates:],
            Variable(scores2.data[:, -self.params.n_rec_updates:])
        )

        # game features loss
        if self.n_features:
            loss_gf = self.loss_fn_gf(output_gf, features.float())
        else:
            loss_gf = 0

        self.register_loss(loss_history, loss_sc, loss_gf)

        return loss_sc, loss_gf


    @staticmethod
    def register_args(parser):
        DQN.register_args(parser)
        parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in MultiHeadAttention")
        parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
        parser.add_argument("--hidden_dim", type=int, default=256, help="Dimension of the transformer layers")
        parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
        parser.add_argument("--pos", type=str, default="learned", help="Type of positional encoding (learned, sin, none)")
        parser.add_argument("--gate", type=str, default="res", help="Gating mechanism (res, gru)")
        parser.add_argument("--identity", type=bool_flag, default=False, help="Use identity map reordering")
        parser.add_argument("--action_dim", type=int, default=0, help="Dimensionality for action embeddings")

    @staticmethod
    def validate_params(params):
        DQN.validate_params(params)
        assert params.num_heads >= 1
        assert params.num_layers >= 1
        assert params.hidden_dim > 0
        assert params.dropout >= 0
        assert params.pos in ["learned", "sin", "none"]
        assert params.gate in ["res", "gru"]
        assert params.action_dim >= 0






