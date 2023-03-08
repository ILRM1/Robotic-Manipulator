import torch
from GPT.training.trainer import Trainer
import torch.nn as nn

class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, timesteps, attention_mask = self.get_batch()
        action_target = torch.clone(actions)
        action_preds = self.transformer.forward(states, actions, timesteps, attention_mask=attention_mask)

        action_target = action_target.reshape(-1, self.action_dim)

        # compute the huber loss
        huber_Loss=nn.HuberLoss(reduction='sum')
        loss=huber_Loss(action_preds, action_target)

        # optimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 0.25)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss.detach().cpu().item()
    
