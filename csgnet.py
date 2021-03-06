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
from resnet_embed import Encoder

N_ITER = 0
TBX = {
    'loss_tr' : 'data/l_train',
    'loss_va' : 'data/l_val',
    'acc_1' : 'data/acc_1',
    'acc_2' : 'data/acc_2',
    'acc_3' : 'data/acc_3',
    'all_acc' : 'data/all_acc',
    'acc_1_val' : 'data/acc_1_val',
    'acc_2_val' : 'data/acc_2_val',
    'acc_3_val' : 'data/acc_3_val',
    'all_acc_val' : 'data/all_acc_val'
}

class DeformDataLoader(data_utils.TensorDataset):
    def __init__(self, dataX, dataY, sample=8, permute=True):
        self.dataX = dataX
        self.dataY = dataY
        self.length = dataX.shape[0]
        self.sample = sample
        self.permute = permute

    def __getitem__(self, idx):
        if self.permute:
            sample_ind = np.random.choice(24, self.sample, replace=False)
        else:
            sample_ind = np.arange(self.sample)
        dataX = self.dataX[idx, sample_ind, ...].flatten()
        return torch.from_numpy(dataX),\
               torch.from_numpy(self.dataY[np.newaxis, idx, 0]),\
               torch.from_numpy(self.dataY[np.newaxis, idx, 1]),\
               torch.from_numpy(self.dataY[np.newaxis, idx, 2])

    def __len__(self):
        return self.length

def load_dataset(x_path, y_path, batch_size=300, num_workers=2, 
                 sample=8, permute=True):
    print("loading dataset")
    stock_data = DeformDataLoader(np.load(x_path),
                                 np.load(y_path),
                                 sample=sample,
                                 permute=permute)
    print("Making dataset loader")
    stock_loader = data_utils.DataLoader(stock_data,
                                         batch_size = batch_size,
                                         shuffle = True,
                                         num_workers = num_workers)
    return stock_loader

class Decoder(nn.Module):
    def __init__(self, num_rnn_layer=1, sample=8, decompose=False):
        super(Decoder, self).__init__()
        self.rnn_layer = num_rnn_layer
        self.sample = sample
        self.decompose = decompose
        # First shape
        self.fc1 = nn.Linear(512*self.sample, 2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.rnn1 = nn.GRU(input_size=2048,
                          hidden_size=2048,
                          num_layers=self.rnn_layer,
                          batch_first=True)
        self.fc2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU(inplace=True)
        self.fcR1 = nn.Linear(2048, 270)
        self.softmax1 = nn.Softmax()
        # Second shape
        self.fc3 = nn.Linear(270, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.rnn2 = nn.GRU(input_size=2048+128,
                          hidden_size=2048,
                          num_layers=self.rnn_layer,
                          batch_first=True)
        self.fc4 = nn.Linear(2048, 2048)
        self.relu4 = nn.ReLU(inplace=True)
        self.fcR2 = nn.Linear(2048, 270)
        self.softmax2 = nn.Softmax()
        # Operation
        self.fc5 = nn.Linear(540, 128)
        self.relu5 = nn.ReLU(inplace=True)
        self.rnn3 = nn.GRU(input_size=2048+128,
                          hidden_size=2048,
                          num_layers=self.rnn_layer,
                          batch_first=True)
        self.fc6 = nn.Linear(2048, 2048)
        self.relu6 = nn.ReLU(inplace=True)
        self.fcR3 = nn.Linear(2048, 720)
        self.softmax3 = nn.Softmax()
        # If decompose
        if self.decompose:
            

    def forward(self, x):
        # First shape
        encode = self.fc1(x)
        encode = self.relu1(encode)
        x1, h1  = self.rnn1(encode[:,None])
        x1 = self.fc2(x1)
        x1 = self.relu2(x1)
        x1 = torch.squeeze(self.fcR1(x1))
        # x1 = self.softmax1(torch.squeeze(x1))
        # Second shape
        h2i = self.fc3(x1)
        h2i = self.relu3(h2i)
        h2i = torch.cat((encode,h2i), dim=1)
        x2, h2  = self.rnn2(h2i[:,None], h1)
        x2 = self.fc4(x2)
        x2= self.relu4(x2)
        x2 = torch.squeeze(self.fcR2(x2))
        # x2 = self.softmax2(torch.squeeze(x2))
        # Operation
        h3i = torch.cat((x1, x2), dim=1)
        h3i = self.fc5(h3i)
        h3i = self.relu5(h3i)
        h3i = torch.cat((encode, h3i), dim=1)
        x3, _  = self.rnn3(h3i[:,None], h2)
        x3 = self.fc6(x3)
        x3 = self.relu6(x3)
        x3 = torch.squeeze(self.fcR3(x3))
        # x3 = self.softmax3(torch.squeeze(x3))
        return x1, x2, x3


def loss_func(output, target):
    return F.cross_entropy(output, target)

def loss3_func(output, target):
    return F.cross_entropy(output, target-1)

def train(dec, train_loader, optimizer, writer, log_interval=1):
    global N_ITER
    dec.train()
    bar = ProgressBar()
    for batch_idx, (data, t1, t2, t3) in bar(enumerate(train_loader)):
        N_ITER += 1
        batch_size = data.size()[0]
        data, t1, t2, t3 = data.cuda(), t1.cuda(), t2.cuda(), t3.cuda()
        data, vt1, vt2, vt3 = Variable(data), Variable(t1), Variable(t2), Variable(t3)
        optimizer.zero_grad()
        out1, out2, out3 = dec(data)
        out1_predict = torch.max(out1.data, 1, keepdim=True)[1]
        out2_predict = torch.max(out2.data, 1, keepdim=True)[1]
        out3_predict = torch.max(out3.data, 1, keepdim=True)[1]
        
        acc1 = torch.eq(out1_predict,t1)
        acc2 = torch.eq(out2_predict,t2)
        acc3 = torch.eq(out3_predict,t3)
        all_acc = float((acc1 * acc2 * acc3).sum())

        loss = loss_func(out1, torch.squeeze(vt1))
        loss += loss_func(out2, torch.squeeze(vt2))
        loss += loss3_func(out3, torch.squeeze(vt3))
        loss.backward()
        optimizer.step()

        # Add scalars to writer
        writer.add_scalar(TBX['loss_tr'], loss.data[0]/batch_size, N_ITER)
        writer.add_scalar(TBX['acc_1'], float(acc1.sum())/batch_size, N_ITER)
        writer.add_scalar(TBX['acc_2'], float(acc2.sum())/batch_size, N_ITER)
        writer.add_scalar(TBX['acc_3'], float(acc3.sum())/batch_size, N_ITER)
        writer.add_scalar(TBX['all_acc'], all_acc/batch_size, N_ITER)

def validate(dec, val_loader, epoch, writer):
    dec.eval()
    test_loss = 0
    acc1_t = 0
    acc2_t = 0
    acc3_t = 0
    all_acc = 0
    for data, t1, t2, t3 in val_loader:
        batch_size = data.size()[0]
        data, t1, t2, t3 = data.cuda(), t1.cuda(), t2.cuda(), t3.cuda()
        data, vt1, vt2, vt3 = Variable(data), Variable(t1), Variable(t2), Variable(t3)
        out1, out2, out3 = dec(data)
        out1_predict = torch.max(out1.data, 1, keepdim=True)[1]
        out2_predict = torch.max(out2.data, 1, keepdim=True)[1]
        out3_predict = torch.max(out3.data, 1, keepdim=True)[1]

        acc1 = torch.eq(out1_predict,t1)
        acc1_t += acc1.sum()
        acc2 = torch.eq(out2_predict,t2)
        acc2_t += acc2.sum()
        acc3 = torch.eq(out3_predict,t3)
        acc3_t += acc3.sum()
        all_acc += (acc1 * acc2 * acc3).sum()

        test_loss += loss_func(out1, torch.squeeze(vt1))
        test_loss += loss_func(out2, torch.squeeze(vt2))
        test_loss += loss_func(out3, torch.squeeze(vt3))

    val_size = len(val_loader.dataset)
    avg_loss = test_loss/val_size
    writer.add_scalar(TBX['loss_va'], avg_loss.data[0], epoch)
    writer.add_scalar(TBX['acc_1_val'], float(acc1_t)/val_size, N_ITER)
    writer.add_scalar(TBX['acc_2_val'], float(acc2_t)/val_size, N_ITER)
    writer.add_scalar(TBX['acc_3_val'], float(acc3_t)/val_size, N_ITER)
    writer.add_scalar(TBX['all_acc_val'], float(all_acc)/val_size, N_ITER)
    return avg_loss.data[0]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Stock data trainer')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for input trianing data, default=64')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to be train, defualt=10')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--name', type=str, metavar='Name',
                        help='name to save in log directory')
    parser.add_argument('--sample_size', type=int, default=8,
                        help="Number of channel samples per model")
    parser.add_argument('--rnn_layer', type=int, default=1,
                        help="Number of RNN layers")
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help="l2 regularization")
    parser.add_argument('--optimizer', type=str, default='adam',
                        help="optimization method")
    parser.add_argument('--permute', type=bool, default=True,
                        help="optimization method")

    args = parser.parse_args()

    CUR_DIR = os.getcwd()
    log_dir = os.path.join(CUR_DIR, 'log', args.name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    train_loader = load_dataset('trX.npy', 'trY.npy', 
                                batch_size=args.batch_size, permute=args.permute)
    val_loader = load_dataset('valX.npy', 'valY.npy', batch_size=args.batch_size)

    decoder = Decoder(num_rnn_layer=args.rnn_layer).cuda()

    if args.optimizer == "adam":
        optimizer = optim.Adam(decoder.parameters(),
                           lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(decoder.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay, nesterov=True)
        drop_interval = args.epochs / 5
        decay = lambda epoch: args.lr * math.pow(0.5, math.floor(epoch/drop_interval))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, decay)
    else:
        raise(ValueError)

    writer = SummaryWriter()
    
    min_loss = float('inf')
    for ep in range(args.epochs):
        is_best = False
        if args.optimizer == 'sgd':
            scheduler.step()
        print("Starting epoch %d"%ep)
        train(decoder, train_loader, optimizer, writer)
        ep_loss = validate(decoder, val_loader, ep, writer)

        is_best = False
        if ep_loss < min_loss:
            is_best = True
            min_loss = ep_loss
        if ep % 50 == 0 or is_best:
            save_checkpoint({
                'epoch': ep,
                'state_dict': decoder.state_dict(),
                'best_acc': ep_loss,
                'optimizer' : optimizer.state_dict(),
                }, is_best,
                filename="{}/checkpoint.{}th.tar".format(log_dir, ep))
