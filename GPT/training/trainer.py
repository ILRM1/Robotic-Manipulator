import numpy as np
import time
import os
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, action_dim, loss_fn=None,scheduler=None):
        self.transformer = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.diagnostics = dict()
        self.action_dim = action_dim
        self.total_it = 0
        self.start_time = time.time()

    def train_iteration(self,group_name, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.transformer.train()
        for _ in tqdm(range(num_steps)):
            self.total_it += 1
            train_loss = self.train_step()
            train_losses.append(train_loss)

        dir_name="model"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        torch.save(
            {
                "model": group_name,
                "epoch": self.total_it,
                "model_state_dict": self.transformer.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "cost": train_loss,
                "description": f"CustomModel-{iter_num}"
            },
            f"{dir_name}/checkpoint-{iter_num}.pt",
        )

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs



