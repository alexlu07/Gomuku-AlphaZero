import torch
import time

from env import Env
from selfplay import SelfplayWrapper
from train import Trainer
from model import Model

def train(save=True, load=True):
    env = SelfplayWrapper()
    trainer = Trainer(env)

    with open("./results/log.txt", "r+") as f:
        if load:
            epoch = int(f.readlines()[-1].split()[0][:-1])
        else:
            f.truncate(0)

    if save: trainer.save_state()
    if load: trainer.load_state(epoch)

    while True:
        start = time.time()
        loss_pi, loss_v, ep_lens, ep_rets, selfplay_time, training_time = trainer.train_one_epoch()

        avg_rets = sum(ep_rets)/len(ep_rets)
        avg_lens = sum(ep_lens)/len(ep_lens)

        duration = time.time() - start

        log = f"{trainer.epoch}: loss={{{loss_pi:.4f}, {loss_v:.4f}}} episodes={{{avg_rets:.4f}, {avg_lens:.4f}}} time={{{duration:.4f}, {selfplay_time:.4f}, {training_time:.4f}}}"
        print(log)

        if save:
            trainer.save_state()
            with open("./results/log.txt", "a") as f:
                f.write(log + "\n")

def watch(a, b):
    env = Env()

    model1 = Model(env.obs_dim, env.act_dim)
    model2 = Model(env.obs_dim, env.act_dim)

    print(model1.state_dict().keys())

    model1.load_state_dict(torch.load(f"./results/weights/{a}.pt")["model"])
    model2.load_state_dict(torch.load(f"./results/weights/{b}.pt")["model"])

    models = [model1, model2]
    player = 0

    done = False
    obs = env.reset()
    env.render()

    while not done:
        act, _, _ = models[player].step(torch.as_tensor(obs, dtype=torch.float32), 
                                legal_actions=torch.tensor(env.legal_actions))
        input()
        obs, _, done = env.step(act)
        env.render()

        player = 1 - player

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