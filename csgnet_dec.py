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
    'acc_1' : 'data/acc_s1',
    'acc_2' : 'data/acc_s2',
    'acc_3' : 'data/acc_d',
    'acc_4' : 'data/acc_rot',
    'all_acc' : 'data/all_acc',
    'acc_1_val' : 'data/acc_s1_val',
    'acc_2_val' : 'data/acc_s2_val',
    'acc_3_val' : 'data/acc_d_val',
    'acc_4_val' : 'data/acc_rot_val',
    'all_acc_val' : 'data/all_acc_val'
}

class DeformDataLoader(data_utils.TensorDataset):
    def __init__(self, dataX, dataY, sample=8, permute=True, angles=12):
        self.dataX = dataX
        self.dataY = dataY
        self.length = dataX.shape[0]
        self.sample = sample
        self.permute = permute
        self.angles = angles

    def __getitem__(self, idx):
        if self.permute:
            sample_ind = np.random.choice(self.angles, self.sample, replace=False)
        else:
            sample_ind = np.arange(self.sample)
        dataX = self.dataX[idx, sample_ind, ...].flatten()
        return torch.from_numpy(dataX),\
               torch.from_numpy(self.dataY[np.newaxis, idx, 0]),\
               torch.from_numpy(self.dataY[np.newaxis, idx, 1]),\
               torch.from_numpy(self.dataY[np.newaxis, idx, 2]),\
               torch.from_numpy(self.dataY[np.newaxis, idx, 3])

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

class RnnUnit(nn.Module):
    def __init__(self, input_dim, num_class, latent_dim=128, rnn_dim=2048, rnn_layer=1):
        super(RnnUnit, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim+rnn_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.rnn = nn.GRU(input_size=rnn_dim+latent_dim,
                          hidden_size=rnn_dim,
                          num_layers=rnn_layer,
                          batch_first=True)
        self.fc2 = nn.Linear(rnn_dim, rnn_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(rnn_dim, num_class)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = torch.unsqueeze(x,1)
        x, _ = self.rnn(x)
        x = self.relu2(self.fc2(x))
        return torch.squeeze(self.fc3(x))


class Decoder(nn.Module):
    def __init__(self, num_shape=270, num_rnn_layer=1, sample=8, rnn_dim=2048, 
                 latent_dim=128, dist_class=20, rot_class=72):
        super(Decoder, self).__init__()
        self.rnn_layer = num_rnn_layer
        self.sample = sample
        # First shape
        self.fc1 = nn.Linear(512*self.sample, rnn_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.rnn1 = nn.GRU(input_size=rnn_dim,
                          hidden_size=rnn_dim,
                          num_layers=self.rnn_layer,
                          batch_first=True)
        self.fc2 = nn.Linear(rnn_dim, rnn_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fcR1 = nn.Linear(rnn_dim, num_shape)      
        # Second shape
        self.decoder1 = RnnUnit(num_shape+rnn_dim, num_shape)
        # distance
        self.decoder2 = RnnUnit(num_shape*2+rnn_dim, dist_class)
        # angles
        self.decoder3 = RnnUnit(num_shape*2+rnn_dim+dist_class, rot_class)
            

    def forward(self, x):
        # First shape
        encode = self.fc1(x)
        encode = self.relu1(encode)
        x0, _  = self.rnn1(encode[:,None])
        x0 = self.relu2(self.fc2(x0))
        x0 = torch.squeeze(self.fcR1(x0))      # [batch x num_shape]
        # Second shape
        x1 = torch.cat((encode,x0), dim=1)
        x1 = self.decoder1(x1)
        # Distance
        x2 = torch.cat((encode, x0, x1), dim=1)
        x2 = self.decoder2(x2)
        # Rotation
        x3 = torch.cat((encode, x0, x1, x2), dim=1)
        x3 = self.decoder3(x3)
        return x0, x1, x2, x3


def loss_func(output, target):
    return F.cross_entropy(output, target)

def train(dec, train_loader, optimizer, writer, log_interval=1):
    global N_ITER
    dec.train()
    bar = ProgressBar()
    for _, (data, t1, t2, t3, t4) in bar(enumerate(train_loader)):
        N_ITER += 1
        batch_size = data.size()[0]
        data, t1, t2, t3, t4 = data.cuda(), t1.cuda(), t2.cuda(), t3.cuda(), t4.cuda()
        data, vt1, vt2, vt3, vt4 = Variable(data), Variable(t1), Variable(t2), Variable(t3), Variable(t4)
        optimizer.zero_grad()
        out1, out2, out3, out4 = dec(data)
        out1_predict = torch.max(out1.data, 1, keepdim=True)[1]
        out2_predict = torch.max(out2.data, 1, keepdim=True)[1]
        out3_predict = torch.max(out3.data, 1, keepdim=True)[1]
        out4_predict = torch.max(out4.data, 1, keepdim=True)[1]
        
        acc1 = torch.eq(out1_predict,t1)
        acc2 = torch.eq(out2_predict,t2)
        acc3 = torch.eq(out3_predict,t3)
        acc4 = torch.eq(out4_predict,t4)
        all_acc = float((acc1 * acc2 * acc3 * acc4).sum())

        loss = loss_func(out1, torch.squeeze(vt1))
        loss += loss_func(out2, torch.squeeze(vt2))
        loss += loss_func(out3, torch.squeeze(vt3))
        loss += loss_func(out4, torch.squeeze(vt4))
        loss.backward()
        optimizer.step()

        # Add scalars to writer
        writer.add_scalar(TBX['loss_tr'], loss.data[0]/batch_size, N_ITER)
        writer.add_scalar(TBX['acc_1'], float(acc1.sum())/batch_size, N_ITER)
        writer.add_scalar(TBX['acc_2'], float(acc2.sum())/batch_size, N_ITER)
        writer.add_scalar(TBX['acc_3'], float(acc3.sum())/batch_size, N_ITER)
        writer.add_scalar(TBX['acc_4'], float(acc4.sum())/batch_size, N_ITER)
        writer.add_scalar(TBX['all_acc'], all_acc/batch_size, N_ITER)

def validate(dec, val_loader, epoch, writer):
    dec.eval()
    test_loss = 0
    acc1_t = 0
    acc2_t = 0
    acc3_t = 0
    acc4_t = 0
    all_acc = 0
    for data, t1, t2, t3, t4 in val_loader:
        data, t1, t2, t3, t4 = data.cuda(), t1.cuda(), t2.cuda(), t3.cuda(), t4.cuda()
        data, vt1, vt2, vt3, vt4 = Variable(data), Variable(t1), Variable(t2), Variable(t3), Variable(t4)
        out1, out2, out3, out4 = dec(data)
        out1_predict = torch.max(out1.data, 1, keepdim=True)[1]
        out2_predict = torch.max(out2.data, 1, keepdim=True)[1]
        out3_predict = torch.max(out3.data, 1, keepdim=True)[1]
        out4_predict = torch.max(out4.data, 1, keepdim=True)[1]

        acc1 = torch.eq(out1_predict,t1)
        acc1_t += acc1.sum()
        acc2 = torch.eq(out2_predict,t2)
        acc2_t += acc2.sum()
        acc3 = torch.eq(out3_predict,t3)
        acc3_t += acc3.sum()
        acc4 = torch.eq(out4_predict,t4)
        acc4_t += acc4.sum()
        all_acc += (acc1 * acc2 * acc3 * acc4).sum()

        test_loss += loss_func(out1, torch.squeeze(vt1))
        test_loss += loss_func(out2, torch.squeeze(vt2))
        test_loss += loss_func(out3, torch.squeeze(vt3))
        test_loss += loss_func(out4, torch.squeeze(vt4))

    val_size = len(val_loader.dataset)
    avg_loss = test_loss/val_size
    writer.add_scalar(TBX['loss_va'], avg_loss.data[0], epoch)
    writer.add_scalar(TBX['acc_1_val'], float(acc1_t)/val_size, N_ITER)
    writer.add_scalar(TBX['acc_2_val'], float(acc2_t)/val_size, N_ITER)
    writer.add_scalar(TBX['acc_3_val'], float(acc3_t)/val_size, N_ITER)
    writer.add_scalar(TBX['acc_4_val'], float(acc4_t)/val_size, N_ITER)
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

    train_loader = load_dataset('trX.npy', 'trY_2.npy', 
                                batch_size=args.batch_size, permute=args.permute)
    val_loader = load_dataset('valX.npy', 'valY_2.npy', batch_size=args.batch_size)

    decoder = Decoder(num_rnn_layer=args.rnn_layer).cuda()

    if args.optimizer == "adam":
        optimizer = optim.Adam(decoder.parameters(),
                           lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(decoder.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay, nesterov=True)
        drop_interval = 20
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
