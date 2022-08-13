import numpy as np

import math
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class Node:
    def __init__(self, parent, p):
        self.parent = parent
        self.children = {}
        self.P = p
        self.Q = 0
        self.N = 0

    def expand(self, pi):
        for act, prob in pi:
            self.children[act] = Node(self, prob)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda c: c[1].UCB_value(c_puct))

    def update(self, leaf_value):
        self.N += 1
        self.Q += (leaf_value - self.Q) / self.N

    def UCB_value(self, c_puct):
        return self.Q + c_puct * self.P * math.sqrt(self.parent.N) / (1+self.N)

    def not_leaf(self):
        return bool(self.children)

class MCTS:
    def __init__(self, model, config, selfplay=False):
        self.root = Node(None, 1.0)
        self.policy = model.step

        self.config = config
        self.selfplay = selfplay

    def simulate(self, env):
        node = self.root
        path = [node]
        while node.not_leaf():
            act, node = node.select(self.config.c_puct)
            env.step(act)

            path.append(node)

        end, winner = env.is_finished()
        if not end:
            pi, value = self.policy(env)
            node.expand(pi)
        else:
            value = winner * env.player

        for i in range(len(path)-1, -1, -1):
            value *= -1
            path[i].update(value)

    def get_mcts_pi(self, env, temp=1e-3):
        for _ in range(self.config.num_simulations):
            self.simulate(copy.deepcopy(env))

        act_visits = [(act, node.N)
                      for act, node in self.root.children.items()]
        actions, visits = zip(*act_visits)

        mcts_pi = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return actions, mcts_pi

    
    def get_action(self, env, temp=1e-3):
        assert env.is_finished()[0] == False

        mcts_pi = np.zeros(env.board_area, dtype="float32")
        acts, pi = self.get_mcts_pi(env, temp)
        mcts_pi[list(acts)] = pi

        if self.selfplay:
            action = np.random.choice(
                acts, 
                p=0.75*pi + 0.25*np.random.dirichlet(0.3*np.ones(len(pi)))
            )
            self.root = self.root.children[action]
        else:
            action = np.random.choice(acts, p=pi)
            self.root = Node(None, 1.0)

        return action, mcts_pi