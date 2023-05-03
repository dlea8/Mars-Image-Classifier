from __future__ import print_function
import argparse
from cProfile import label
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from MarsDataset import MarsDataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1)
        self.conv2 = nn.Conv2d(3, 1, 3, 1)
        self.fc1 = nn.Linear(12321, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # loss_values = []
    running_loss = 0.0
    running_accuracy = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
            running_loss =+ loss.item()
            # running_accuracy =+ 100. * batch_idx / len(train_loader)

    return [100. * correct/len(train_loader.dataset), running_loss]


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return [100. * correct/len(test_loader.dataset), test_loss]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device: ", device)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # mars_dataset = MarsDataset("/Users/blakebollinger/Library/Mobile Documents/com~apple~CloudDocs/LSU/Spring 2023/Math 4997/Group Project/Mars-Image-Classifier/mars-dataset/image-labels.csv", "/Users/blakebollinger/Library/Mobile Documents/com~apple~CloudDocs/LSU/Spring 2023/Math 4997/Group Project/Mars-Image-Classifier/mars-dataset/images")

    transform = transforms.Compose([transforms.ToTensor()])

    mars_dataset = datasets.ImageFolder('../mars-dataset/images', transform=transform)

    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                    transform=transform)
    train_loader = torch.utils.data.DataLoader(mars_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(mars_dataset, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    training_loss_values = []
    training_accuracy_values = []
    test_accuracy_values = []
    test_loss_values = []

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, 15):
        test_returns = test(model, device, test_loader)
        train_returns = train(args, model, device, train_loader, optimizer, epoch)

        training_accuracy_values.append(train_returns[0])
        training_loss_values.append(train_returns[1])

        test_accuracy_values.append(test_returns[0])
        test_loss_values.append(test_returns[1])
        scheduler.step()


    #  trianing graphs'
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(training_loss_values, label='Train Loss')
    plt.plot(test_loss_values, label='test Loss')
    plt.legend()
    plt.show()

    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Percent Accuracy')
    plt.plot(training_accuracy_values, label='Train Accuracy')
    plt.plot(test_accuracy_values, label='Test Accuracy')
    plt.ylim(0, 100)
    plt.legend()
    plt.show()



    # testing graphs
    # plt.title("Test Set Accuracy")
    # plt.xlabel('epoch')
    # plt.ylabel('percent accuracy')
    # plt.plot(test_accuracy_values)
    # plt.ylim(0, 100)
    # plt.show()


    # plt.title("Test Set Loss")
    # plt.xlabel('epoch')
    # plt.ylabel('losses')
    # plt.plot(test_loss_values)
    # plt.show()





    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")

    main()