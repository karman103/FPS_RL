from base import DQN
from dtqn_aly import DTQN_aly
import torch
from torch.autograd import Variable
class DTQNAgent(DQN):
    DQNModuleClass = DTQN_aly
    def __init__(self, params):
        super(DTQNAgent, self).__init__(params)
        # IDK ????
    def reset(self):
        # prev_state is only used for evaluation, so has a batch size of 1
        self.prev_state = self.init_state_e

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

        screens = screens.view(batch_size, seq_len * self.params.n_fm,
                               *self.screen_shape[1:])
        # I DONT GET IT
        output_sc1, output_gf1 = self.module(
            screens[:, :-self.params.n_fm, :, :],
            [variables[:, -2, i] for i in range(self.params.n_variables)]
        )
        output_sc2, output_gf2 = self.module(
            screens[:, self.params.n_fm:, :, :],
            [variables[:, -1, i] for i in range(self.params.n_variables)]
        )

        # compute scores
        mask = torch.ByteTensor(output_sc1.size()).fill_(0)
        for i in range(batch_size):
            mask[i, int(actions[i, -1])] = 1
        scores1 = output_sc1.masked_select(self.get_var(mask))
        scores2 = rewards[:, -1] + (
            self.params.gamma * output_sc2.max(1)[0] * (1 - isfinal[:, -1])
        )

        # dqn loss
        loss_sc = self.loss_fn_sc(scores1, Variable(scores2.data))

        # game features loss
        loss_gf = 0
        if self.n_features:
            loss_gf += self.loss_fn_gf(output_gf1, features[:, -2].float())
            loss_gf += self.loss_fn_gf(output_gf2, features[:, -1].float())

        self.register_loss(loss_history, loss_sc, loss_gf)

        return loss_sc, loss_gf