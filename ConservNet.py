# coding=utf-8
import argparse
import os
import numpy as np
import scipy as sc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import (
    DataLoader,
)  # (testset, batch_size=4,shuffle=False, num_workers=4)
from torch.utils.data.dataset import TensorDataset

import pickle

import tracemalloc
import distutils
import distutils.util
 
from src.utils import *
from src.system import *
from src.net import *

def str2bool(v):
    return bool(distutils.util.strtobool(v))

parser = argparse.ArgumentParser(description="Pytorch ConservNet Training")

parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=10000, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

parser.add_argument(
    "--lr",
    "--learning-rate",
    default=5e-5,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0,
    type=float,
    metavar="W",
    help="weight decay (default: 0.01)",
    dest="weight_decay",
)
parser.add_argument(
    "--model", "--model", default="Con", type=str, help="simulation data type : Con(servNet), Siam"
)
parser.add_argument(
    "--system", default="S1", type=str, help="simulation sytem, S1, S2, S3"
)
parser.add_argument(
    "--spreader", default="L2", type=str, help="spreader, L1, L2, L8"
)

parser.add_argument("--iter", default=10, type=int, help="iter num")

parser.add_argument("--n", default=10, type=int, help="group num")
parser.add_argument("--m", default=200, type=int, help="data num")

parser.add_argument("--Q", default=1., type=float, help="Spreader constant")
parser.add_argument("--R", default=1., type=float, help="Noise norm")
parser.add_argument("--beta", default=1., type=float, help="Variance term constant")
parser.add_argument("--noise", default=0., type=float, help="noise strength")

parser.add_argument(
    "--indicator", default="", type=str, help="Additional specification for file name."
)
parser.add_argument("--seed", default=42, type=int, help="Random seed for torch and numpy")

         
    
def train(model, train_loader, optimizer, plugin, epoch, Q, beta):
    train_losses = AverageMeter("TrainLoss", ":.4e")
    #Q = Q*max(1-epoch/(1000), 0.1)
    for image, label in train_loader:
        label = label.cuda()
        image = image.cuda()
        d1 = model(image) 
        d2 = model(image + plugin.generate(image).cuda())
        train_loss = torch.var(d1) + beta * torch.abs(Q - torch.var(d2, dim=0))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_losses.update(train_loss.item(), image.shape[0])
    return train_losses.avg

def test(model, test_loader, plugin, epoch, Q, beta):
    test_losses = AverageMeter("TestLoss", ":.4e")
    mean_var = AverageMeter("MeanVar", ":.4e") 
    
    image = test_loader.dataset.tensors[0]
    label = test_loader.dataset.tensors[1]

    pred = DCN(model(image.cuda()).squeeze(-1))
    slope, intercept, r_value, p_value, std_err = sc.stats.linregress(pred, DCN(label))
    
    for image, label in test_loader:
        label = label.cuda()
        image = image.cuda()
        d1 = model(image) 
        d2 = model(image + plugin.generate(image).cuda())
        test_loss = torch.var(d1) + beta * torch.abs(Q - torch.var(d2))
        test_losses.update(test_loss.item(), image.shape[0])
        mean_var.update(torch.std(d1 * slope).item())
        
    return test_losses.avg, r_value, mean_var.avg

# transform Kepler problem's data from Cartesian to polar coordinate
def transform(data):
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    vx = data[:, 2]
    vy = data[:, 3]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    r_dot = (x*vx+y*vy)/r
    theta_dot = ((x*vy-y*vx)/x**2)*np.cos(theta)**2
    data[:,0] = r
    data[:,1] = r_dot
    data[:,2] = theta
    data[:,3] = theta_dot
    return data

def main():
    tracemalloc.start()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parameter check

    print(f'system : {args.system}')
    print(f'spreader : {args.spreader}')
    print(f'iter : {args.iter}')
    print(f'n : {args.n}')
    print(f'm : {args.m}')
    print(f'Q : {args.Q}')
    print(f'R : {args.R}')
    print(f'beta : {args.beta}')
    print(f'noise : {args.noise}')
    print(f'indicator : {args.indicator}')

    system_dict = {'S1': system_S1, 'S2': system_S2, 'S3': system_S3, 'P1': system_P1, 'P2': system_P2, 'P3' : system_P3}
    len_dict = {'S1': (4, 0), 'S2':(3, 0), 'S3': (4, 0), 'P1': (2, 0), 'P2': (4, 0), 'P3': (4, 0)}

    if args.system != 'P3':
        formula_len = len_dict[args.system][0]
        noise_len = len_dict[args.system][1]

        system_name = system_dict[args.system]
        rule_name = args.model + '_' + args.system
        batch_num = args.n
        # batch_size = int(2000 / batch_num) * 2
        batch_size = args.m * 2
        total_size = args.n * batch_size
        batch_num = int(total_size / batch_size)
        train_ratio = 0.5
        noise = args.noise
    else:
        formula_len= 4
        noise_len = 0
        plugin_name = plugin_P3
        rule_name = 'Con_P3' 
        total_size = 818 # 818 datapoints total
        batch_size = 818 
        batch_num = int(total_size/batch_size)
        train_ratio = 0.9

    generator = DataGen(system_name, batch_size, batch_num)
    file_name = rule_name + '_L' + str(formula_len) + '_N' + str(noise_len) +'_B' + str(batch_num) + '_n' + str(noise)

    if not os.path.isfile('./data/' + file_name + '_train.pkl'):
        generator.run(file_name, total_size, batch_size, train_ratio, noise_strength=noise)

    # Loader

    with open('./data/' + file_name + '_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('./data/' + file_name + '_test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    noise_var = args.noise
    train_shape = torch.FloatTensor(train_data['Image']).shape
    test_shape = torch.FloatTensor(test_data['Image']).shape
    tmax = torch.ones(formula_len + noise_len)
    if args.system == 'P1':
        tmax = torch.FloatTensor([10., 10.])
    elif args.system == 'P2':
        tmax = torch.FloatTensor([10., 10., 1., 1.])
    elif args.system == 'P3':
        tmax = torch.FloatTensor([1., 1., 10., 10.])

    rtheta = False  # change this when polar coordinate is needed
    if rtheta:
        train_data['Image'] = transform(train_data['Image'])
        test_data['Image'] = transform(test_data['Image'])
        tmax = torch.ones(formula_len + noise_len)

    if args.system == 'P2':
        train_data = TensorDataset(torch.FloatTensor(train_data['Image']) / tmax + noise_var * torch.randn(*train_shape),
                                torch.FloatTensor(train_data['Label'])[:, 0])

        test_data = TensorDataset(torch.FloatTensor(test_data['Image']) / tmax + noise_var * torch.randn(*train_shape),
                                torch.FloatTensor(test_data['Label'])[:, 0])
    else:
        train_data = TensorDataset(torch.FloatTensor(train_data['Image']) / tmax + noise_var * torch.randn(*train_shape),
                            torch.FloatTensor(train_data['Label']))

        test_data = TensorDataset(torch.FloatTensor(test_data['Image']) / tmax + noise_var * torch.randn(*train_shape),
                                torch.FloatTensor(test_data['Label']))

    train_loader = DataLoader(
                train_data,
                batch_size=int(train_ratio * batch_size),
                shuffle=False,
                pin_memory=True,
                num_workers=args.workers
            )
    test_loader = DataLoader(
                test_data,
                batch_size=int((1 - train_ratio) * batch_size),
                shuffle=False,
                pin_memory=True,
                num_workers=args.workers
            )

    for i in range(formula_len + noise_len):
        print('x{} : min = {}, max = {}'.format(i, min(train_data.tensors[0][:,i]), max(train_data.tensors[0][:,i])))
    #print(f'C : min = {min(train_data.tensors[1][:,0])}, max = {max(train_data.tensors[1][:,0])}')
    print(f'C : min = {min(train_data.tensors[1])}, max = {max(train_data.tensors[1])}')

    # Spreader
   
    D_in = formula_len + noise_len
    D_hidden = 320
    D_out = 1
    cfg_clf = [D_in, D_hidden, D_hidden, D_hidden, D_hidden, D_out]
    model_list = []
    indicator = args.indicator
    
    for iter in range(args.iter):
        model = ConservNet(cfg_clf, 'mlp', 1).cuda()
        train_loss_list = []
        test_loss_list = []
        mv_list = []
        corr_list = []
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        best_loss = np.inf
        plugin = spreader('L2', args.R)
        Q = args.Q
        beta = args.beta
        best_model = None
       
        for epoch in range(0, args.epochs):
            train_loss = train(model, train_loader, optimizer, plugin, epoch, Q, beta)
            test_loss, corr, mean_var = test(model, test_loader, plugin, epoch, Q, beta)
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            mv_list.append(mean_var)
            corr_list.append(np.abs(corr))

            if is_best:
                best_model = model
            
        model_list.append({
                        "epoch": epoch,
                        "model_state_dict": best_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": test_loss,
                        "MV": mean_var,
                        "best_loss": best_loss,
                        "train_loss_list": train_loss_list,
                        "test_loss_list": test_loss_list,
                        "mv_list": mv_list,
                        "corr_list": corr_list})
    
    with open('./result/' + file_name + indicator + '.pkl', 'wb') as f:
        pickle.dump(model_list, f)

if __name__ == "__main__":
    print("started!")  # For test
    main()
