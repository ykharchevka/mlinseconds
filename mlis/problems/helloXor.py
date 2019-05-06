# HelloXor is a HelloWorld of Machine Learning.
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        sm.SolutionManager.print_hint("Hint[1]: Xor can not be learned with only one layer")
        self.model = nn.Sequential(
            nn.Linear(2, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model.forward(x)

    def calc_loss(self, output, target):
        loss_fn = nn.BCELoss()
        loss = loss_fn(output, target)
        return loss

    def calc_predict(self, output):
        predict = output.round()
        return predict

class Solution():
    def __init__(self):
        self = self

    def create_model(self, input_size, output_size):
        model = SolutionModel(input_size, output_size)
        for param in model.parameters():
            nn.init.uniform_(param, -1.0, +1.0)
        return model

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        prev_loss = None
        # Put model in train mode
        model.train()
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                break
            sm.SolutionManager.print_hint("Hint[2]: Learning rate is too small", step)
            optimizer = optim.RMSprop(model.parameters(), lr=.017, alpha=.95, eps=1e-08, weight_decay=.0, momentum=.0)
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            if correct == total:
                print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, prev_loss))
                break
            # calculate loss
            loss = model.calc_loss(output, target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            prev_loss = loss.item()
            step += 1
        return step

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 100
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self):
        data = torch.FloatTensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
            ])
        target = torch.FloatTensor([
            [0.0],
            [1.0],
            [1.0],
            [0.0]
            ])
        return (data, target)

    def create_case_data(self, case):
        data, target = self.create_data()
        return sm.CaseData(case, Limits(), (data, target), (data, target))

class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
