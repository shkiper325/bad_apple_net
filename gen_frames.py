import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
from tqdm import tqdm

DUMP_X_FILE = 'dump_X.npy'
MODEL_PATH = 'models/550'
OUT_FRAMES_DIR = 'out_frames'
TH = 0.1

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

if USE_CUDA:
    print('Using CUDA')
else:
    print('Using CPU')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense_1 = nn.Linear(in_features=6572, out_features=512)
        self.dense_1_bn = nn.BatchNorm1d(512)
        self.dense_2 = nn.Linear(in_features=512, out_features=512)
        self.dense_2_bn = nn.BatchNorm1d(512)
        self.dense_3 = nn.Linear(in_features=512, out_features=512)
        self.dense_3_bn = nn.BatchNorm1d(512)

        self.us_1 = nn.Upsample(size=(5, 5))
        self.conv_1_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv_1_1_bn = nn.BatchNorm2d(512)
        self.conv_1_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv_1_2_bn = nn.BatchNorm2d(256)

        self.us_2 = nn.Upsample(size=(10, 10))
        self.conv_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_2_1_bn = nn.BatchNorm2d(256)
        self.conv_2_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv_2_2_bn = nn.BatchNorm2d(128)

        self.us_3 = nn.Upsample(size=(20, 20))
        self.conv_3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_3_1_bn = nn.BatchNorm2d(128)
        self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_3_2_bn = nn.BatchNorm2d(128)

        self.us_4 = nn.Upsample(size=(40, 40))
        self.conv_4_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_4_1_bn = nn.BatchNorm2d(128)
        self.conv_4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_4_2_bn = nn.BatchNorm2d(128)

        self.us_5 = nn.Upsample(size=(80, 80))
        self.conv_5_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_5_1_bn = nn.BatchNorm2d(128)
        self.conv_5_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_5_2_bn = nn.BatchNorm2d(128)

        self.conv_out = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1)
        
        if USE_CUDA:
            self.cuda()
            
    def forward(self, x):
        x = F.relu(self.dense_1_bn(self.dense_1(x)))
        x = F.relu(self.dense_2_bn(self.dense_2(x)))
        x = F.relu(self.dense_3_bn(self.dense_3(x)))

        x = torch.reshape(x, (-1, 512, 1, 1))

        x = self.us_1(x)
        x = F.relu(self.conv_1_1_bn(self.conv_1_1(x)))
        x = F.relu(self.conv_1_2_bn(self.conv_1_2(x)))

        x = self.us_2(x)
        x = F.relu(self.conv_2_1_bn(self.conv_2_1(x)))
        x = F.relu(self.conv_2_2_bn(self.conv_2_2(x)))

        x = self.us_3(x)
        x = F.relu(self.conv_3_1_bn(self.conv_3_1(x)))
        x = F.relu(self.conv_3_2_bn(self.conv_3_2(x)))

        x = self.us_4(x)
        x = F.relu(self.conv_4_1_bn(self.conv_4_1(x)))
        x = F.relu(self.conv_4_2_bn(self.conv_4_2(x)))

        x = self.us_5(x)
        x = F.relu(self.conv_5_1_bn(self.conv_5_1(x)))
        x = F.relu(self.conv_5_2_bn(self.conv_5_2(x)))

        x = torch.sigmoid(self.conv_out(x))
        
        return x

def main():
    if not os.path.exists(OUT_FRAMES_DIR):
        os.mkdir(OUT_FRAMES_DIR)

    X = np.load(DUMP_X_FILE)
    N = X.shape[0]

    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH)['weights'])
    model.eval()

    frames = []
    for i in tqdm(range(N)):
        x = FloatTensor(np.expand_dims(X[i], axis=0))
        frame = model(x).detach().cpu().numpy()[0, 0]
        frame = frame * 255

        frames.append(frame)

    print(np.mean(frames))

    for i in range(N):
        cv2.imwrite(os.path.join(OUT_FRAMES_DIR, str(i).zfill(4) + '.png'), frames[i])

if __name__ == '__main__':
    main()
