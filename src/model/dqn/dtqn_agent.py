from .base import DQN
from .dtqn_aly import DTQN_aly
import torch
from torch.autograd import Variable
from ...utils import bool_flag

class DTQNAgent(DQN):
    DQNModuleClass = DTQN_aly
    def __init__(self, params):
        super(DTQNAgent, self).__init__(params)
        # IDK ????
    # def reset(self):
    #     # prev_state is only used for evaluation, so has a batch size of 1
    #     self.prev_state = self.init_state_e

    def f_eval(self, last_states):

        screens, variables = self.prepare_f_eval_args(last_states)

        output = self.module( # CHANGE THE PARAMETERS
            screens.view(1, self.hist_size, *self.screen_shape),
            [variables[:, i].contiguous().view(1, self.hist_size)
                for i in range(self.params.n_variables)],
        )

        return output

    def f_train(self, screens, variables, features, actions, rewards, isfinal,
                loss_history=None):

        screens, variables, features, actions, rewards, isfinal = \
            self.prepare_f_train_args(screens, variables, features,
                                      actions, rewards, isfinal)

        batch_size = self.params.batch_size
        seq_len = self.hist_size + 1

        # I DONT GET IT
        output_sc, output_gf = self.module(
            screens,
            [variables[:, :, i] for i in range(self.params.n_variables)],
        )

        # compute scores
        mask = torch.BoolTensor(output_sc.size()).fill_(0)
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
        parser.add_argument("--n_rec_updates", type=int, default=1,
                            help="Number of updates to perform")
        parser.add_argument("--remember", type=bool_flag, default=True,
                            help="Remember the whole sequence")

    @staticmethod
    def validate_params(params):
        DQN.validate_params(params)
        assert params.n_rec_updates >= 1      