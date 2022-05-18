# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.optim as optim

# Torchvision
import torchvision
import torchvision.transforms as transforms

# OS
import os
import argparse
import pickle

def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model(lf_num):
    autoencoder = Autoencoder(lf_num)
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


class Autoencoder(nn.Module):
    def __init__(self, lfnum):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, lfnum, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(lfnum, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


parser = argparse.ArgumentParser(description="Train Autoencoder")
parser.add_argument("--data", "-D", metavar='DT', type=str, default='CIFAR10')
parser.add_argument("--iter", "-I", metavar='IT', type=int, default=5)
parser.add_argument("--epoch", "-E", metavar='EP', type=int, default=2000)
parser.add_argument("--lfnum", "-L", metavar='LF', type=int, default=48, help='The number of filter at the latent representation')
parser.add_argument("--alpha", "-A", metavar='AL', type=float, default=1.)

args = parser.parse_args()


# Set random seed for reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Load data
transform = transforms.Compose(
    [transforms.ToTensor(), ])

if args.data == 'CIFAR10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
elif args.data == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)


alpha, beta = args.alpha, 1 - args.alpha
print(alpha, beta)

# Define an stat storage
result_dict = {f'model{i}': {'weight': None, 'TrainLoss1': [], 'TrainLoss2': [], 'TestLoss1': [], 'TestLoss2': []} for i in range(args.iter)}

for i in range(args.iter):
    # Create model
    autoencoder = create_model(args.lfnum)

    # Define an optimizer and criterion
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=5e-4)
    BestTestLoss = 1000.

    for epoch in range(args.epoch):
        best_changed = False
        running_loss_1, running_loss_2 = 0., 0.
        for (inputs, _) in trainloader:
            inputs = get_torch_vars(inputs,)

            # ============ Forward ============
            encoded_1, outputs_1 = autoencoder(inputs)
            loss_1 = criterion(outputs_1, inputs)
            encoded_2, outputs_2 = autoencoder(outputs_1)
            loss_2 = criterion(outputs_2, inputs)
            loss = alpha * loss_1 + beta * loss_2
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss_1 += loss_1.data
            running_loss_2 += loss_2.data
            
        TrainLoss1 = running_loss_1 / len(trainloader)
        TrainLoss2 = running_loss_2 / len(trainloader)
        result_dict[f'model{i}']['TrainLoss1'] = TrainLoss1
        result_dict[f'model{i}']['TrainLoss2'] = TrainLoss2
        
        running_loss_1, running_loss_2 = 0., 0.
        with torch.no_grad():
            for (inputs, _) in testloader:
                inputs = get_torch_vars(inputs)

                # ============ Forward ============
                encoded_1, outputs_1 = autoencoder(inputs)
                loss_1 = criterion(outputs_1, inputs)
                encoded_2, outputs_2 = autoencoder(outputs_1)
                loss_2 = criterion(outputs_2, inputs)
                loss = alpha * loss_1 + beta * loss_2
                # ============ Logging ============
                running_loss_1 += loss_1.data
                running_loss_2 += loss_2.data

        TestLoss1 = running_loss_1 / len(testloader)
        TestLoss2 = running_loss_2 / len(testloader)
        result_dict[f'model{i}']['TestLoss1'] = TrainLoss1
        result_dict[f'model{i}']['TestLoss2'] = TrainLoss2
        if TestLoss1 < BestTestLoss:
            result_dict[f'model{i}']['weight'] = autoencoder.state_dict()
            best_changed = True
            BestTestLoss = TestLoss1

        print(f'[{epoch}]')
        print(f'TrainLoss1 : {TrainLoss1}, TrainLoss_2 : {TrainLoss2}')
        print(f' TestLoss1 : {TestLoss1},  TrainLoss_2 : {TestLoss2}')
        print(f'BestTestLoss : {BestTestLoss}' + (' best changed' if best_changed else ''))

        with open(f'./result/result_{args.data}_I{args.iter}_E{args.epoch}_L{args.lfnum}_A{args.alpha}.pkl', 'wb') as f:
            pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
