import ray
import numpy as np

from model import Model
from mcts import MCTS

@ray.remote
class Selfplay:
    def __init__(self, env, config):
        self.env = env
        self.buffer = buffer
        self.config = config

        self.model = Model(env.board_area, self.config.selfplay_device)
        
    def run(self, weights):
        self.model.load(weights)

        states, mcts_pi, rewards = self.run_selfplay()
        augmented_data = self.augment_data(zip(states, mcts_pi, rewards))
        
        return augmented_data

    def run_selfplay(self, epoch):
        states = []
        mcts_pi = []
        rewards = []

        mcts = MCTS(self.model, self.config.c_puct, self.config.num_simulations, True)
        self.env.reset()
        while True:
            act, pi = mcts.get_action(self.env, self.config.softmax_temp(epoch))

            states.append(self.env.get_observation())
            mcts_pi.append(pi)
            rewards.append(self.env.player)

            self.env.step(act)

            end, winner = self.env.is_finished()
            if end:
                rewards = winner * np.array(rewards)

                return states, mcts_pi, rewards

    def augment_data(self, data):
        p = True
        augmented_data = []
        for state, mcts_prob, reward in data:
            for r in range(4):
                if p:
                    print(state)
                    print(mcts_prob.reshape(self.env.board_size, self.env.board_size))
                aug_state = np.rot90(state, r, (1, 2))
                aug_mcts_prob = np.rot90(
                    mcts_prob.reshape(self.env.board_size, self.env.board_size), r)
                augmented_data.append((aug_state, aug_mcts_prob.flatten(), reward))

                aug_state = np.flip(aug_state, axis=2)
                aug_mcts_prob = np.fliplr(mcts_prob)
                augmented_data.append((aug_state, aug_mcts_prob.flatten(), reward))

        return augmented_data