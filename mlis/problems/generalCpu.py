# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils.gridsearch import GridSearch

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_depth):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        sm.SolutionManager.print_hint("Hint[1]: Explore more deep neural networks")
        hidden_layers = []
        for i in range(hidden_depth):
            hidden_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU()
            ])
        self.model = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.LeakyReLU(),
            *hidden_layers,
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )
        for param in self.model.parameters():
            nn.init.uniform_(param, -1.0, +1.0)

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
        self.lr = 0.01
        self.lr_grid = [.01, .1, 1., 10., 100.]
        self.hidden_size = 5
        self.hidden_size_grid = [1, 2, 3, 4, 5, 6, 7, 8]
        self.hidden_depth = 4
        self.hidden_depth_grid = [1, 2, 3, 4, 5, 6, 7, 8]
        self.grid_search = GridSearch(self).set_enabled(False)  # TODO: set to True, if grid search needed

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self.hidden_size, self.hidden_depth)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                break
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=self.lr,
                alpha=.95,
                weight_decay=.0,
                momentum=.0,
                eps=1e-08,
            )
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            sm.SolutionManager.print_hint("Hint[2]: Explore other activation functions", step)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            if correct == total:
                break
            # calculate loss
            sm.SolutionManager.print_hint("Hint[3]: Explore other loss functions", step)
            loss = model.calc_loss(output, target)
            self.grid_search.log_step_value('loss', loss.item(), step)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            self.print_stats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step
    
    def print_stats(self, step, loss, correct, total):
        if step % 1000 == 0:
            print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, loss.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size  # 2**input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):  # 8...128
            for j in range(input_size):  # 3...7
                input_bit = (i>>j)&1  # (i // 2**j) & 1 -> 1 iff (i // 2**j) == 1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data,
                target.view(-1, 1)  # e.g.: size [128] -> [128, 1]
                )

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(
            case,
            Limits(),
            (data, target),
            (data, target)
        ).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
