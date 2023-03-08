import time
import cv2
import numpy as np
import wandb
import argparse
import pickle
import torch
from GPT.models.transformer import Transformer
from GPT.training.seq_trainer import SequenceTrainer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from tqdm import tqdm

#customize dataloader
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, length, max_ep_len, state_mean, state_std, start_time):
        self.img_dir = img_dir
        self.length=length
        self.max_ep_len=max_ep_len
        self.state_mean=state_mean.transpose((2,0,1))
        self.state_std=state_std.transpose((2,0,1))
        self.start_time=start_time

    def __len__(self):
        return self.length

    # stack 10 images
    def __getitem__(self, idx):
        color_path = self.img_dir+str(idx)

        # normalize images
        color = read_image(color_path+"/"+str(self.start_time[idx])+".png")
        color = ((color/255-self.state_mean)/self.state_std).unsqueeze(0)

        for i in range(self.start_time[idx]+1,self.start_time[idx]+self.max_ep_len-1):
            c = read_image(color_path+"/"+str(i)+".png")
            c = ((c/255-self.state_mean)/self.state_std).unsqueeze(0)
            color = torch.cat((color,c),dim=0)

        return color, idx

def experiment(variant):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant['log_to_wandb']

    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{env_name}-{dataset}'

    img_num=10
    max_ep_len = img_num+1

    img_dir="roll_resize/img"

    # load the first timesteps of appeared ball
    with open(f"start_time.pkl", 'rb') as f:
        start_time = pickle.load(f)
    # load the trajectories label
    with open(f"label.pkl", 'rb') as f:
        trajectories = pickle.load(f)

    # compute mean and standard deviation of images
    sum_state=0
    total=0
    for i in range(len(start_time)):
        for j in range(start_time[i],start_time[i]+img_num):
            img=cv2.imread(img_dir+str(i)+"/"+str(j)+".png")
            sum_state+=img/255
            total+=1
    state_mean = sum_state / total

    state_std = 0
    for i in range(len(start_time)):
        for j in range(start_time[i], start_time[i] + img_num):
            img = cv2.imread(img_dir + str(i) + "/" + str(j) + ".png")
            state_std += abs(img / 255 - state_mean) ** 2
    state_std = np.sqrt(state_std / total) + 1e-6

    # with open(f'data.pkl', 'rb') as f:
    #     data=pickle.load(f)
    # state_mean, state_std=data['state_mean'],data['state_std']

    states, depth, actions, traj_lens= [], [], [], []
    for i, path in enumerate(trajectories):
        traj_lens.append(path.shape[0])

    # compute mean and standard deiviaition of trajectories
    actions = np.concatenate(trajectories, axis=0)
    action_mean, action_std = np.mean(actions, axis=0), np.std(actions, axis=0)+1e-6

    action_dim = trajectories.shape[1]*trajectories.shape[2]
    state_dim = state_mean.shape
    s = {"state_mean": state_mean, "state_std": state_std, "action_mean": action_mean, "action_std": action_std,"max_ep_len":max_ep_len,"action_dim":action_dim}
    with open(f'data.pkl', 'wb') as f:
        pickle.dump(s, f)

    # load transformer model
    model = Transformer(
        state_dim=state_dim,
        act_dim=action_dim,
        channel=state_dim[2],
        max_length=max_ep_len,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=3 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout']
    )

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print('=' * 50)

    batch_size = variant['batch_size']

    train_dataset = CustomImageDataset(img_dir=img_dir,
                                       length=len(traj_lens),
                                       max_ep_len=max_ep_len,
                                       state_mean=state_mean,
                                       state_std=state_std,
                                       start_time=start_time,
                                       )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # get batch of images and trajectories converted to tensor
    def get_batch():
        for j in train_dataloader:
            color,idx=j
            break

        a, timesteps, mask = [], [], []
        for i in idx.numpy():
            a.append(trajectories[i])

            timesteps.append(np.arange(max_ep_len).reshape(1, -1))

            # normalize trajectories
            a[-1]= (a[-1] - action_mean) / action_std
            mask.append(np.ones(max_ep_len).reshape(1, -1))

        s = color.to(dtype=torch.float32, device=device)
        a=np.array(a).reshape(batch_size,-1)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=device).unsqueeze(1)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, timesteps, mask

    model = model.to(device=device)
    model = torch.nn.DataParallel(model)

    warmup_steps = variant['warmup_steps']

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    # checkpoint = torch.load('model5/checkpoint-10.pt')
    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # checkpoint_epoch = checkpoint["epoch"]
    # checkpoint_description = checkpoint["description"]
    # print(checkpoint_description)

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        action_dim=action_dim
    )

    if log_to_wandb:
        wandb.init(
            project='GPT',
            entity='lny'
        )
        # wandb.watch(model)  # wandb has some bug
        wandb.config.update(args)
    for iter in tqdm(range(variant['max_iters'])):
        outputs = trainer.train_iteration(group_name=group_name, num_steps=variant['num_steps_per_iter'], iter_num=iter + 1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='sim')
    parser.add_argument('--dataset', type=str, default='roll')
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    args = parser.parse_args()
    experiment(variant=vars(args))