import torch
from torchvision.models import resnet18
import numpy as np
import os
from resnet_embed import Encoder
import cv2
import multiprocessing as mp
from torch.autograd import Variable
import tqdm

DATA_DIR = os.path.join(os.getcwd(), 'data')
encoder = Encoder(resnet18(pretrained=True), 24).cuda()

def encode_folder(folder):
    folder_mat = []
    folder = os.path.join(DATA_DIR, folder)
    for fname in os.listdir(folder):
        if '.png' in fname:
            f_path = os.path.join(folder, fname)
            im = cv2.imread(f_path)
            im = im.astype(np.float32)
            im = np.rollaxis(im, 2)
            folder_mat.append(im)
    if folder_mat is not []:
        folder_ims = np.stack(folder_mat)
        inputs = Variable(torch.from_numpy(folder_ims)).cuda()
        embed = encoder(inputs).cpu().numpy()
    save_path = os.path.join(folder, 'embed.npy')
    np.save(save_path, embed.astype(np.float32))
    return embed

def gather_embed():
    X = []
    Y = []
    for f in os.listdir(DATA_DIR):
        subdir = os.path.join(DATA_DIR, f)
        embed_name = os.path.join(subdir, 'embed.npy')
        label_name = os.path.join(subdir, 'label.npy')
        if os.path.exists(embed_name):
            X.append(np.load(embed_name))
        if os.path.exists(label_name):
            Y.append(np.load(label_name))
    print("Saving data")
    print(np.shape(X))
    print(np.shape(Y))
    X = np.stack(X)
    Y = np.stack(Y)
    n = np.shape(X)[0]
    div = int(n*0.8)
    np.save('trX.npy', X[:div])
    np.save('trY.npy', Y[:div])
    np.save('valX.npy', X[div:])
    np.save('valY.npy', Y[div:])

if __name__ == "__main__":
    for f in tqdm.tqdm(os.listdir(DATA_DIR)):
        encode_folder(f)
    gather_embed()