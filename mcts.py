import numpy as np
import torch
import copy


class TreeNode(object):
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}
        self.n_visit = 0
        self.Q = 0
        self.u = 0
        self.P = prior_prob

    def expand(self, actions):
        for action, prob in actions.items():
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c):
        return max(self.children.items(), key=lambda item: item[1].node_value(c))

    def node_value(self, c):
        self.u = (c * self.P * np.sqrt(self.parent.n_visit) / (1 + self.n_visit))
        return self.Q + self.u

    def update(self, leaf_value):
        if self.parent:
            self.parent.update(-leaf_value)
        self.n_visit += 1
        self.Q += (leaf_value - self.Q) / self.n_visit

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTS(object):
    def __init__(self, network, c=5, n_rollout=1000, use_gpu=False):
        self.root = TreeNode(None, 1.0)
        self.network = network
        self.c = c
        self.n_rollout = n_rollout
        self.use_gpu = use_gpu

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.root = TreeNode(None, 1.0)

    def rollout(self, board_state):
        node = self.root

        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c)
            board_state.do_move(action)

        avail_actions = board_state.availables
        current_state = torch.from_numpy(np.ascontiguousarray(board_state.current_state().reshape(
            -1, 4, 6, 6)))
        if self.use_gpu:
            log_action_probs, leaf_value = self.network(current_state.cuda().float())
            action_probs = np.exp(log_action_probs.data.cpu().numpy().flatten())
        else:
            log_action_probs, leaf_value = self.network(current_state.float())
            action_probs = np.exp(log_action_probs.data.numpy().flatten())

        leaf_value = leaf_value.data[0][0]
        end, winner = board_state.game_end()

        if not end:
            act_probs = {}
            for action in avail_actions:
                act_probs[action] = action_probs[action]
            node.expand(act_probs)
        else:
            if winner == -1:  # tie
                leaf_value = 0
            else:
                leaf_value = 1.0 if winner == board_state.get_current_player() else -1.0

        node.update(-leaf_value)

    def get_action(self, board_state, temp=0.001, ret_prob=False, self_play=False):
        """
        :param temp: level of exploration
        :return:
        """
        avail_actions = board_state.availables

        if len(avail_actions) <= 0:
            print("board is full")
            return

        for n in range(self.n_rollout):
            state_copy = copy.deepcopy(board_state)
            self.rollout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.n_visit)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)

        act_probs = 1.0/temp * np.log(np.array(visits) + 1e-10)
        act_probs = np.exp(act_probs - np.max(act_probs))
        act_probs /= np.sum(act_probs)

        probs = np.zeros(6 * 6)
        probs[list(acts)] = act_probs

        if self_play:
            move = np.random.choice(
                acts,
                p=0.75 * act_probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(act_probs)))
            )
            # update the root node and reuse the search tree
            if move in self.root.children:
                self.root = self.root.children[move]
                self.root.parent = None
            else:
                self.root = TreeNode(None, 1.0)
        else:
            move = np.random.choice(acts, p=act_probs)
            self.root = TreeNode(None, 1.0)

        if ret_prob:
            return move, probs
        else:
            return move



