# from random import gammavariate
# import torch
# from torch.optim import Adam
# import numpy as np
# import scipy
# import time

# from model import Model

# class Trainer:
#     def __init__(self, env, timesteps_per_batch=4000, train_pi_iters=80, train_v_iters=80, 
#                  pi_lr=3e-4, vf_lr=1e-3, gamma=0.99, lam=0.97, clip_ratio=0.2):
#         self.env = env
#         self.obs_dim = env.obs_dim
#         self.act_dim = env.act_dim

#         self.epoch = 0

#         self.timesteps_per_batch = timesteps_per_batch
#         self.train_pi_iters = train_pi_iters
#         self.train_v_iters = train_v_iters
#         self.pi_lr = pi_lr
#         self.vf_lr = vf_lr
#         self.gamma = gamma
#         self.lam = lam
#         self.clip_ratio = clip_ratio

#         self.history = GameHistory(self.obs_dim, self.act_dim, 
#             self.timesteps_per_batch + env.max_steps, self.gamma, self.lam)

#         self.model = Model(self.obs_dim, self.act_dim)
#         self.pi_optimizer = Adam(self.model.pi.parameters(), lr=self.pi_lr)
#         self.vf_optimizer = Adam(self.model.vf.parameters(), lr=self.vf_lr)

#     def train_one_epoch(self):
#         ep_lens = []
#         ep_rets = []

#         start = time.time()

#         timesteps = 0
#         while timesteps < self.timesteps_per_batch:
#             t, game_len, game_ret = self.play_game()
            
#             ep_lens.append(game_len)
#             ep_rets.append(game_ret)
#             timesteps += t

#         selfplay_time = time.time() - start
#         start = time.time()

#         data = self.history.get()

#         pi_loss_old = self.get_pi_loss(data).item()
#         v_loss_old = self.get_vf_loss(data).item()

#         for i in range(self.train_pi_iters):
#             self.pi_optimizer.zero_grad()
#             pi_loss = self.get_pi_loss(data)
#             pi_loss.backward()
#             self.pi_optimizer.step()

#         for i in range(self.train_v_iters):
#             self.vf_optimizer.zero_grad()
#             vf_loss = self.get_vf_loss(data)
#             vf_loss.backward()
#             self.vf_optimizer.step()

#         training_time = time.time() - start

#         self.epoch += 1

#         return pi_loss_old, v_loss_old, ep_lens, ep_rets, selfplay_time, training_time


#     def play_game(self):
#         t = 0
#         game_len = 0
#         game_ret = 0

#         obs, done = self.env.reset(self.epoch), False
#         while not done:
#             act, val, lgp = self.model.step(torch.as_tensor(obs, dtype=torch.float32), 
#                                             legal_actions=torch.tensor(self.env.legal_actions))

#             next_obs, rew, done = self.env.step(act)

#             self.history.store(obs, act, rew, val, lgp)

#             obs = next_obs

#             t += 1
#             game_len += 1
#             game_ret += rew
        
#         self.history.finish_path()
#         return t, game_len, game_ret

#     def get_pi_loss(self, data):
#         obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

#         pi = self.model.actor_dist(obs)
#         logp = pi.log_prob(act)
#         ratio = torch.exp(logp - logp_old)
#         clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
#         loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

#         return loss_pi
    
#     def get_vf_loss(self, data):
#         obs, ret = data['obs'], data['ret']
#         return ((self.model.critic(obs) - ret)**2).mean()

#     def save_state(self):
#         torch.save({
#             "model": self.model.state_dict(),
#             "pi_optimizer": self.pi_optimizer.state_dict(),
#             "vf_optimizer": self.vf_optimizer.state_dict(),
#             "epoch": self.epoch,
#         }, f"./results/weights/{self.epoch}.pt")

#     def load_state(self, e):
#         checkpoint = torch.load(f"./results/weights/{e}.pt")

#         self.model.load_state_dict(checkpoint["model"])
#         self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer"])
#         self.vf_optimizer.load_state_dict(checkpoint["vf_optimizer"])
#         self.epoch = checkpoint["epoch"]

# class GameHistory:
    
#     def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
#         self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
#         self.act_buf = np.zeros(size, dtype=np.float32)
#         self.adv_buf = np.zeros(size, dtype=np.float32)
#         self.rew_buf = np.zeros(size, dtype=np.float32)
#         self.ret_buf = np.zeros(size, dtype=np.float32)
#         self.val_buf = np.zeros(size, dtype=np.float32)
#         self.logp_buf = np.zeros(size, dtype=np.float32)
#         self.gamma, self.lam = gamma, lam
#         self.ptr, self.path_start_idx, self.max_size = 0, 0, size

#     def store(self, obs, act, rew, val, logp):
#         assert self.ptr < self.max_size     # buffer has to have room so you can store

#         self.obs_buf[self.ptr] = obs
#         self.act_buf[self.ptr] = act
#         self.rew_buf[self.ptr] = rew
#         self.val_buf[self.ptr] = val
#         self.logp_buf[self.ptr] = logp
#         self.ptr += 1

#     def finish_path(self, last_val=0):
#         path_slice = slice(self.path_start_idx, self.ptr)
#         rews = np.append(self.rew_buf[path_slice], last_val)
#         vals = np.append(self.val_buf[path_slice], last_val)
        
#         # the next two lines implement GAE-Lambda advantage calculation
#         deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
#         self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        
#         # the next line computes rewards-to-go, to be targets for the value function
#         self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

#         self.path_start_idx = self.ptr

#     def get(self):
#         # the next two lines implement the advantage normalization trick
#         adv_mean = np.mean(self.adv_buf)
#         adv_std = np.std(self.adv_buf)
#         self.adv_buf = (self.adv_buf - adv_mean) / adv_std

#         data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
#                     adv=self.adv_buf, logp = self.logp_buf)
#         result = {k: torch.as_tensor(v[:self.ptr], dtype=torch.float32) for k,v in data.items()}

#         self.ptr, self.path_start_idx = 0, 0

#         return result

#     def discount_cumsum(self, x, discount):
#         return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]