import torch
from torch.optim import Adam
import numpy as np
import time
from collections import deque
import ray
import copy

from model import Model
from selfplay import Selfplay

class Trainer:
    def __init__(self, env, config):
        self.env = env
        self.board_size = self.env.board_size
        self.board_area = self.env.board_area

        self.config = config

        self.epoch = 0
        
        self.selfplay_actors = [
            Selfplay.remote(copy.deepcopy(self.env), self.config) for i in range(self.config.selfplay_actors)]
        self.data_buffer = deque(max_len=config.buffer_size)

        self.model = Model(self.board_area, device=self.train_device)
        self.optimizer = Adam(self.model.parameters(), weight_decay=config.l2_const)

    # def train_one_epoch(self):
    #     ep_lens = []
    #     ep_rets = []

    #     start = time.time()

    #     timesteps = 0
    #     while timesteps < self.timesteps_per_batch:
    #         t, game_len, game_ret = self.play_game()
            
    #         ep_lens.append(game_len)
    #         ep_rets.append(game_ret)
    #         timesteps += t

    #     selfplay_time = time.time() - start
    #     start = time.time()

    #     data = self.history.get()

    #     pi_loss_old = self.get_pi_loss(data).item()
    #     v_loss_old = self.get_vf_loss(data).item()

    #     for i in range(self.train_pi_iters):
    #         self.pi_optimizer.zero_grad()
    #         pi_loss = self.get_pi_loss(data)
    #         pi_loss.backward()
    #         self.pi_optimizer.step()

    #     for i in range(self.train_v_iters):
    #         self.vf_optimizer.zero_grad()
    #         vf_loss = self.get_vf_loss(data)
    #         vf_loss.backward()
    #         self.vf_optimizer.step()

    #     training_time = time.time() - start

    #     self.epoch += 1

    #     return pi_loss_old, v_loss_old, ep_lens, ep_rets, selfplay_time, training_time

                
    def launch_selfplay_jobs(self):
        w = ray.put(self.model.weights())
        data = [actor.run.remote(w, self.epoch) for actor in self.selfplay_actors]
        for d in data:
            self.data_buffer.extend(ray.get(d))

    # def get_pi_loss(self, data):
    #     obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    #     pi = self.model.actor_dist(obs)
    #     logp = pi.log_prob(act)
    #     ratio = torch.exp(logp - logp_old)
    #     clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
    #     loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    #     return loss_pi
        
    
    # def get_vf_loss(self, data):
    #     obs, ret = data['obs'], data['ret']
    #     return ((self.model.critic(obs) - ret)**2).mean()

    def save_state(self):
        torch.save({
            "model": self.model.weights(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }, f"./results/weights/{self.epoch}.pt")

    def load_state(self, e):
        checkpoint = torch.load(f"./results/weights/{e}.pt")

        self.model.load(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]