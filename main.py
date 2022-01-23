import collections

import numpy as np
import random

import torch

from game import Board, Game
from network import Net
from mcts import MCTS
from pure_mcts import MCTSPlayer
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict



use_gpu = True

class Training():
    def __init__(self):
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 0.002
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_rollout = 400  # num of simulations for each move
        self.c = 5
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = collections.deque(maxlen=5000)
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000

        # start training from a new policy-value net
        self.net = Net(self.board_width, self.board_height)
        if use_gpu:
            self.net = self.net.to(torch.device('cuda'))
        self.optimizer = optim.Adam(self.net.parameters(),
                                    weight_decay=0.0001)
        self.mcts_player = MCTS(self.net, self.c, self.n_rollout, True)

    def run(self):
        for i in range(self.game_batch_num):
            print(str(i) + "th game")
            self.collect_selfplay_data()
            if len(self.data_buffer) >= self.batch_size:
                self.policy_update()

                if i % 20 == 0:
                    win_num, loss_num, tie_num = self.policy_evaluate()
                    print("win: {}, loss: {}, tie: {}".format(win_num, loss_num, tie_num))



    def collect_selfplay_data(self):
        winner, data = self.game.start_self_play(self.mcts_player, temp=self.temp)
        data = self.data_augment(data)
        self.data_buffer.extend(data)

    def data_augment(self, data):
        """augment the data set by rotation and flipping
                extend_data: [(state, mcts_prob, winner_z), ..., ...]
                """
        extend_data = []
        states = data['states']
        mcts_probs = data['mcts_probs']
        winners_z = data['winners_z']

        for i in range(len(states)):
            state, mcts_prob, winner = states[i], mcts_probs[i], winners_z[i]
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        if use_gpu:
            state_batch = torch.FloatTensor(np.array(state_batch)).cuda()
            mcts_probs_batch = torch.FloatTensor(np.array(mcts_probs_batch)).cuda()
            winner_batch = torch.FloatTensor(np.array(winner_batch)).cuda()
        else:
            state_batch = torch.FloatTensor(np.array(state_batch))
            mcts_probs_batch = torch.FloatTensor(np.array(mcts_probs_batch))
            winner_batch = torch.FloatTensor(np.array(winner_batch))

        self.optimizer.zero_grad()

        for group in self.optimizer.param_groups:
            group['lr'] = self.learn_rate * self.lr_multiplier

        action_prob, value = self.net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch * action_prob, 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        new_action_prob, _ = self.net(state_batch)

        if use_gpu:
            action_prob = np.exp(action_prob.data.cpu().numpy())
            new_action_prob = np.exp(new_action_prob.data.cpu().numpy())
        else:
            action_prob = np.exp(action_prob.data.numpy())
            new_action_prob = np.exp(new_action_prob.data.numpy())

        kl = np.mean(np.sum(action_prob * (np.log(action_prob + 1e-10) - np.log(new_action_prob + 1e-10)), axis=1))

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

    def policy_evaluate(self):
        mcts_player = MCTS(self.net, self.c, self.n_rollout, True)
        pure_mcts_player = MCTSPlayer(5, 1000)  # pure mcts player
        win_cnt = defaultdict(int)
        for i in range(10):
            winner = self.game.start_play(mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        # win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / 10
        '''
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))
        '''

        return win_cnt[1], win_cnt[2], win_cnt[-1]

train = Training()
train.run()




