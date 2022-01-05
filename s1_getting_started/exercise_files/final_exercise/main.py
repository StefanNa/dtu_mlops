import argparse
import sys

import torch
from torch import optim, nn

from data import mnist, Dataset_mnist
from model import MyAwesomeModel
import numpy as np

def _power_of_2(num):
    """Check if number is power of 2"""

    cond = np.log2(int(num))

    if np.ceil(cond) != np.floor(cond):
        raise argparse.ArgumentTypeError("Argument must be a power of 2")

    return int(num)

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self,PATH="../../../data/corruptmnist/"):
        ##get data
        self.PATH=PATH
        
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )

        parser.add_argument(
            "command", choices=["train", "evaluate"], help="Subcommand to run"
        )

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument("--c",'-checkpoint',type=str, default=None,help="checkpoint")
        parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
        parser.add_argument("--e", type=int, default=10, help="epoch")
        parser.add_argument("--b", type=_power_of_2, default=64, help="batch size")

        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        load=args.c
        epochs=args.e
        lr=args.lr
        batchsize=args.b

        train_data= mnist(self.PATH,train=True)
        trainset=Dataset_mnist(train_data)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)

        # TODO: Implement training loop here
   
        model = MyAwesomeModel()

        if load is None:
            True
        else:
            state_dict = torch.load(load)
            model.load_state_dict(state_dict)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses=[]

        for e in range(epochs):
            running_loss=0
            model.train()
            for images, labels in trainloader:
                optimizer.zero_grad()
        
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            train_losses.append(running_loss)
            model.eval()
            print(running_loss)
            if e%3==0:
                torch.save(model.state_dict(), 'checkpoint.pth')
                # torch.save(model.state_dict(), 'checkpoint'+str(e)+'.pth')

        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        # parser = argparse.ArgumentParser(description='Training arguments')
        # parser.add_argument('load_model_from', default="")
        # # add any additional argument that you want
        # args = parser.parse_args(sys.argv[2:])
        # print(args)
        
        test_data= mnist(self.PATH,train=False)
        testset=Dataset_mnist(test_data)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        state_dict = torch.load('checkpoint.pth')
        model.load_state_dict(state_dict)

        # torch.load(args.load_model_from)
        # _, test_set = mnist()

        with torch.no_grad():
            # set model to evaluation mode
            model.eval()
        accuracies=[]

        for images, labels in testloader:
            log_ps = model(images)
            ps = torch.exp(log_ps)
            pred=torch.max(ps,1)[1]
            correct= pred==labels.view(*pred.shape)
            accuracy=correct.type(torch.FloatTensor).mean()
            accuracies.append(accuracy.item()*100)
        print(f'Accuracy: {np.mean(accuracies)}%')


if __name__ == '__main__':
    
    TrainOREvaluate()
    # TOE.train()
    # TOE.evaluate()