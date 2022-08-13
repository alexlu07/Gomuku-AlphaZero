import random
from tabnanny import check
import ray
import time
import copy
import torch
from torch.optim import Adam
import numpy as np
from collections import deque

from model import Model
from selfplay import Selfplay
from env import Env
from config import Config

class Trainer:
    def __init__(self, env: Env, config: Config):
        self.env = env
        self.board_size = self.env.board_size
        self.board_area = self.env.board_area

        self.config = config

        self.epoch = 0
        
        self.selfplay_actors = [
            Selfplay.remote(copy.deepcopy(self.env), self.config) for i in range(self.config.num_actors)]
        self.data_buffer = deque(maxlen=config.buffer_size)

        self.model = Model(self.board_area, device=self.config.train_device)
        self.optimizer = Adam(self.model.parameters(), weight_decay=config.l2_const)

    def train_one_epoch(self):
        start = time.time()

        ep_len = self.launch_selfplay_jobs()

        selfplay_time = time.time() - start
        start = time.time()

        while len(self.data_buffer) < self.config.batch_size:
            print("Running selfplay again: not enough data")
            self.launch_selfplay_jobs()
        data = random.sample(self.data_buffer, self.config.batch_size)
        state_batch = torch.as_tensor(np.array([i[0] for i in data]), dtype=torch.float32)
        mcts_pi_batch = torch.as_tensor(np.array([i[1] for i in data]), dtype=torch.float32)
        reward_batch = torch.as_tensor([i[2] for i in data], dtype=torch.float32)

        old_log_pi, old_v = self.model.batch_step(state_batch)

        for i in range(self.config.update_steps):
            self.optimizer.zero_grad()
            self.update_optimizer_lr()

            log_pi, v = self.model.batch_step(state_batch)

            loss_pi = -torch.mean(torch.sum(mcts_pi_batch*log_pi, dim=1))
            loss_v = ((v - reward_batch)**2).mean()
            loss = loss_pi + loss_v

            loss.backward()
            self.optimizer.step()

            entropy = -torch.mean(torch.sum(torch.exp(log_pi) * log_pi, dim=1)).item()

            kl = torch.mean(torch.sum(
                torch.exp(old_log_pi) * (old_log_pi - log_pi), 
                dim=1)).item()

            if kl > self.config.kl_targ * 4:
                break

        if kl > self.config.kl_targ * 2 and self.config.lr_multiplier > 0.1:
            self.config.lr_multiplier /= 1.5
        elif kl < self.config.kl_targ / 2 and self.config.lr_multiplier < 10:
            self.config.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             torch.var(reward_batch - old_v.flatten()) /
                             torch.var(reward_batch)).item()
        explained_var_new = (1 -
                             torch.var(reward_batch - v.flatten()) /
                             torch.var(reward_batch)).item()


        training_time = time.time() - start

        self.epoch += 1

        return ep_len, kl, self.config.lr_multiplier, loss.item(), entropy, explained_var_old, explained_var_new, selfplay_time, training_time

    def launch_selfplay_jobs(self):
        ep_lens = []
        w = ray.put(self.model.weights())
        data = [actor.run.remote(w, self.epoch) for actor in self.selfplay_actors]
        for d, l in ray.get(data):
            ep_lens.append(l)
            self.data_buffer.extend(d)

        return sum(ep_lens) / self.config.num_actors

    def update_optimizer_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr * self.config.lr_multiplier

    def save_state(self):
        torch.save({
            "model": self.model.weights(),
            "optimizer": self.optimizer.state_dict(),
            "lr_mult": self.config.lr_multiplier,
            "buffer": self.data_buffer,
            "epoch": self.epoch,
        }, f"./results/weights/{self.epoch}.pt")

    def load_state(self, e):
        checkpoint = torch.load(f"./results/weights/{e}.pt")

        self.model.load(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            self.config.lr_multiplier = checkpoint["lr_mult"]
            self.data_buffer = checkpoint["buffer"]
        except:
            print("no lr_mult or data_buffer")
        self.epoch = checkpoint["epoch"]