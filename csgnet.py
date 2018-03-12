import argparse
import math
import os
import shutil

import numpy as np
import torch
from progressbar import ProgressBar
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data as data_utils
from torchvision.models import resnet18

N_ITER = 0

class Encoder(nn.Module):
    def __init__(self, model):
        self.model = model
        self.target_layer = self.model._modules.get('avgpool')
        self.embedding = torch.zeros(512)
        self.hook = self.target_layer.register_forward_hook(self._pullHook)
        
    def _pullHook(self, m, i, o):
        self.embedding.copy_(o.data)
    
    def get_embedding(self, x):
        self.model(x)
        return self.embedding

class Decoder_Block(nn.Module):
    def __init__(self, num_rnn_layer=1):
        super(Decoder_Block, self).__init__()
        self.fc1 = nn.Linear(512*24, 2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(360, 128)
        self.relu2 = nn.ReLU(inplace=True)
        self.rnn = nn.GRU(input_size=2048+128,
                          hidden_size=2048,
                          num_layers=1)
        self.fc3 = nn.Linear(2048, 2048)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2048, 360)
        self.softmax = nn.Softmax()

    def forward(self, x, hidden):
        x = self.fc1(x)
        x = self.relu1(x)
        h = self.fc2(hidden)
        h = self.relu2(h)
        x = torch.cat((x,h))
        x, _  = self.rnn(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        output = self.softmax(x)
        return output

class Decoder(nn.Module):
    def __init__(self, block, hidden_size):
        self.hidden_size = hidden_size
        self.block1 = block(num_layers=1)
        
    def forward(self, x, hidden):
        out1 = self.block1(x, hidden)
        out2 = self.block1(x, out1)
        out3 = self.block1(x, out2)
        return out1, out2, out3

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

def loss_func(output, target):
    return nn.CrossEntropyLoss(output, target)

def train(enc, dec, train_loader, optimizer, log_interval, writer, loss_tr):
    global N_ITER
    dec.train()
    bar = ProgressBar()
    for batch_idx, (data, target) in bar(enumerate(train_loader)):
        N_ITER += 1
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        out1, out2, out3 = dec(data)

        loss = loss_func(out1, target[:,0])
        loss += loss_func(out2, target[:,1])
        loss += loss_func(out3, target[:,2])
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            writer.add_scalar(loss_tr, loss.data[0], N_ITER)

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Stock data trainer')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for input trianing data, default=64')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to be train, defualt=10')
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--name', type=str, metavar='Name',
                        help='name to save in log directory')
    args = parser.parse_args()

    CUR_DIR = os.getcwd()
    log_dir = os.path.join(CUR_DIR, 'log', args.name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    encoder = Encoder(resnet18(pretrained=True)).cuda()
    decoder = Decoder(Decoder_Block, 360).cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)

    writer = SummaryWriter()
    loss_tr = os.path.join(log_dir,'data','l_tr')
    loss_va = os.path.join(log_dir,'data','l_va')
    min_loss = float('inf')
    for ep in range(args.epochs):
        print("Starting epoch %d"%ep)
        train(model, train_loader, optimizer, args.batch_size, writer, loss_tr)
        ep_loss = validate(model, val_loader, ep, writer, loss_va)

        is_best = False
        if ep_loss < min_loss:
            is_best = True
            min_loss = ep_loss
        save_checkpoint({
                'epoch': ep,
                'state_dict': model.state_dict(),
                'best_acc': ep_loss,
                'optimizer' : optimizer.state_dict(),
                }, is_best,
                filename="{}/checkpoint.{}th.tar".format(log_dir, ep))
