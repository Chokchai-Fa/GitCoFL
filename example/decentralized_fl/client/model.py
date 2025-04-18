
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


def load_data_set(path: str):
    transform = transforms.Compose( # transform is from torchvision (only for image)
        [transforms.ToTensor(), # image to tensor --> divide by 255
        transforms.Resize((32, 32))])

    batch_size = 32

    trainvalset = torchvision.datasets.CIFAR10(root=path, train=True, download=False, transform=transform)
    trainset, valset = torch.utils.data.random_split(trainvalset, [40000, 10000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainset.__len__(), valset.__len__(), testset.__len__()

    return trainloader, testloader, valloader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels = 3, output channels = 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces spatial size: 64 x 16 x 16
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128 channels
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces spatial size: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256 channels
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces spatial size: 256 x 4 x 4
            
            nn.Flatten(),  # Flatten the tensor for fully connected layers
            nn.Linear(256*4*4, 1024),  # Input size 256*4*4, output size 1024
            nn.ReLU(),
            nn.Linear(1024, 512),  # Input size 1024, output size 512
            nn.ReLU(),
            nn.Linear(512, 10)  # Final layer for 10 output classes
        )
        
    def forward(self, xb):
        return self.network(xb)