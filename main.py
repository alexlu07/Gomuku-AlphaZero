import torch
import time

from env import Env
from mcts import MCTS
from train import Trainer
from model import Model
from config import Config

def train(save=True, load=True):
    env = Env()
    config = Config()
    trainer = Trainer(env, config)

    with open("./results/log.txt", "r+") as f:
        if load:
            epoch = int(f.readlines()[-1].split()[0][:-1])
            trainer.load_state(epoch)
        else:
            if save:
                f.truncate(0)

    while True:
        ep_len, kl, lr_mult, loss, entropy, EV_old, EV_new, selfplay_time, training_time = trainer.train_one_epoch()

        log = f"{trainer.epoch}: (ep_len: {int(ep_len)}, kl: {kl:.4f}, lr_mult: {lr_mult:.4f}, loss: {loss:.4f}, entropy: {entropy:.4f}, EV_old: {EV_old:.4f}, EV_new: {EV_new:.4f}, sp_time: {selfplay_time:.4f}, train_time: {training_time:.4f})"
        print(log)

        if save:
            trainer.save_state()
            with open("./results/log.txt", "a") as f:
                f.write(log + "\n")

def watch(a, b):
    env = Env()

    model1 = Model(env.board_area)
    model2 = Model(env.board_area)

    model1.load_state_dict(torch.load(f"./results/weights/{a}.pt")["model"])
    model2.load_state_dict(torch.load(f"./results/weights/{b}.pt")["model"])

    player1 = MCTS(model1, config)
    player2 = MCTS(model2, config)

    players = [player1, player2]
    player = 0

    done = False
    env.reset()
    env.render()

    while not done:
        act, pi = players[player].get_action(env)
        input()

        env.step(act)
        env.render()

        player = 1-player

        done, winner = env.is_finished()

def play(epoch, first=True):
    env = Env()

    model = Model(env.obs_dim, env.act_dim)

    model.load_state_dict(torch.load(f"./results/weights/{epoch}.pt")["model"])

    person = 0 if first else 1
    player = 0

    done = False
    obs = env.reset()
    env.render()

    while not done:
        if person == player:
            _, act = env.human_input_to_action()
        else:
            time.sleep(1)
            act, _, _ = model.step(torch.as_tensor(obs, dtype=torch.float32), 
                                legal_actions=torch.tensor(env.legal_actions))
    
        obs, _, done = env.step(act)
        env.render()

        player = 1 - player

if __name__ == "__main__":
    train()
    # watch(302, 302)
    # play(302)