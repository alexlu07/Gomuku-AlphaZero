import ray
import numpy as np

from model import Model
from mcts import MCTS
from env import Env
from config import Config

@ray.remote
class Selfplay:
    def __init__(self, env: Env, config: Config):
        self.env = env
        self.config = config

        self.model = Model(env.board_area, self.config.selfplay_device)
        
    def run(self, weights, epoch):
        self.model.load(weights)

        states, mcts_pi, rewards = self.run_selfplay(epoch)
        augmented_data = self.augment_data(zip(states, mcts_pi, rewards))

        game_len = len(states)
        
        return augmented_data, game_len

    def run_selfplay(self, epoch):
        t = 0
        states = []
        mcts_pi = []
        rewards = []

        mcts = MCTS(self.model, self.config, True)
        self.env.reset()
        while True:
            act, pi = mcts.get_action(self.env, self.config.softmax_temp(epoch))

            states.append(self.env.get_observation())
            mcts_pi.append(pi)
            rewards.append(self.env.player)

            self.env.step(act)

            end, winner = self.env.is_finished()

            t += 1
            if end:
                rewards = winner * np.array(rewards)

                return states, mcts_pi, rewards

    def augment_data(self, data):
        augmented_data = []
        for state, mcts_pi, reward in list(data):
            for r in range(4):
                aug_state = np.rot90(state, r, (1, 2))
                aug_mcts_pi = np.rot90(
                    mcts_pi.reshape(self.env.board_size, self.env.board_size), r)
                augmented_data.append((aug_state, aug_mcts_pi.flatten(), reward))

                aug_state = np.flip(aug_state, axis=2)
                aug_mcts_pi = np.fliplr(aug_mcts_pi)
                augmented_data.append((aug_state, aug_mcts_pi.flatten(), reward))

        return augmented_data