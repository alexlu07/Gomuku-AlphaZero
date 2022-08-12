import torch

class Config:
    def __init__(self):
        self.lr = 2e-3
        self.lr_multiplier = 1
        self.num_simulations = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.selfplay_actors = 2
        self.update_steps = 5
        self.kl_targ = 0.02
        self.l2_const = 1e-4

        self.selfplay_device = torch.device('cpu')
        self.train_device = torch.device('cpu')
    
    def softmax_temp(self, epoch):
        if epoch < 2000:
            return 1.0
        elif epoch < 4000:
            return 0.5
        else:
            return 0.25