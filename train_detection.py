from detection_net import CNN
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.io import read_image
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn as nn

# customize pytorch dataloader
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, data, length):
        self.img_dir = img_dir
        self.length=length
        self.state_mean=data['state_mean'].transpose((2,0,1))
        self.state_std=data['state_std'].transpose((2,0,1))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = read_image(self.img_dir+str(idx)+".png")
        img=((img/255-self.state_mean)/self.state_std)

        la= 1 if idx<20000 else 0.

        return img, la

length = 40000
img_dir='cnn_roll/'

# compute mean and standard deiviation of images
sum_state=0
for i in range(length):
    img=cv2.imread(img_dir+str(i)+".png")
    sum_state+=img/255

state_mean = sum_state / length

state_std=0
for i in range(length):
    img = cv2.imread(img_dir+str(i)+".png")
    state_std+=abs(img/255-state_mean)**2
state_std = np.sqrt(state_std/length) +1e-6

s = {"state_mean": state_mean, "state_std": state_std}
with open(f'cnn_data.pkl', 'wb') as f:
    pickle.dump(s, f)

with open(f'cnn_data.pkl', 'rb') as f:
    data=pickle.load(f)

train = CustomImageDataset(img_dir=img_dir, length=length, data=data)
train_dataloader = DataLoader(train, batch_size=64, shuffle=True)

model = CNN().to('cuda')
model = torch.nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
huber_Loss = nn.HuberLoss(reduction='sum')

# train
for i in tqdm(range(100000),desc="cnn"):

    for j in train_dataloader:
        color, label = j
        break

    # predict labels
    pred_label=model.forward(color.float()).view(-1)

    # compute huber loss
    loss = huber_Loss(pred_label, label.to('cuda').float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(),"cnn_model.pt")