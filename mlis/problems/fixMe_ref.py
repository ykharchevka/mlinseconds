# There are solution for generalCpu in 10 steps.
#
# BUT unfortunetly, someone made mistake and disabled batch normalization.
# See "FIX ME"
#
# You can not simple fix mistake, you can change only activation function at this point
import numpy as np
import math
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

# Note: activation function should be element independent
# See: check_independence method
class MyActivation:
    @staticmethod
    def apply(x):
        global gamma, beta
        x = x.clamp(min=0)
        mu = x.sum().div(x.numel())
        sigma_2 = x.add(-mu).pow(2).sum().div(x.numel())
        x = x.add(-mu).div(sigma_2.add(1e-310).pow(0.5))
        x = x.mul(gamma).add(beta)
        return x

###
###
### Don't change code after this line
###
###
class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.solution = solution
        self.do_norm = solution.do_norm
        layers_size = [input_size] + [solution.hidden_size] * solution.layers_number + [output_size]
        self.linears = nn.ModuleList([nn.Linear(a, b) for a, b in zip(layers_size, layers_size[1:])])
        if self.do_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(a, track_running_stats=False) for a in layers_size[1:]])

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.do_norm:
                x = self.batch_norms[i](x)
            if i != len(self.linears) - 1:
                x = MyActivation.apply(x)
            else:
                x = torch.sigmoid(x)
        return x

    def calc_loss(self, output, target):
        loss = nn.BCELoss()
        return loss(output, target)

    def calc_predict(self, output):
        predict = output.round()
        return predict


class Solution():
    def __init__(self):
        self.gamma = 1.
        self.gamma_grid = [1., 1., 1.]  # np.linspace(1.001, 1.001, 5)
        self.beta = 0.
        self.beta_grid = [0., 0., 0.]
        global gamma, beta
        gamma = self.gamma
        beta = self.beta
        self.learning_rate = 0.05
        self.momentum = 0.8
        self.layers_number = 3
        self.hidden_size = 30
        # FIX ME:) But you can change only activation function
        self.do_norm = False
        self.layers_number = 8
        # self.learning_rate_grid = [0.04, 0.05, 0.06]
        # self.momentum_grid = [0.799, 0.8, 0.801]
        # self.layers_number_grid = [1,2,3,4,5,6,7,8]
        # self.hidden_size_grid = [20, 30, 40]
        # self.do_norm_grid = [True, False]
        self.iter = 0
        self.iter_number = 10  # 100
        self.grid_search = gs.GridSearch(self).set_enabled(False)
        self.grid_search_counter = 0
        self.grid_search_size = eval('*'.join([str(len(v)) for k, v in self.__dict__.items() if k.endswith('_grid')]))

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # activation should be independent
    def check_independence(self):
        ind_size = 100
        for i in range(ind_size + 1):
            x = torch.FloatTensor(ind_size).uniform_(-10, 10)
            same = MyActivation.apply(x)[:i] == MyActivation.apply(x)[:i]
            assert same.long().sum() == i, "Independent function only"

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        self.check_independence()
        step = 0
        loss = 999.
        # Put model in train mode
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                step = math.inf
                break
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
            if total == correct:
                break
            # calculate loss
            loss = model.calc_loss(output, target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            # self.print_stats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        if self.grid_search.enabled:
            # self.grid_search.add_result('step', step)
            self.grid_search.add_result('loss', loss)
            if self.iter == self.iter_number - 1:
                print(self.grid_search.choice_str, self.grid_search.get_stats('loss'))
            self.grid_search_counter += 1
            print('{:>8} / {:>8}'.format(self.grid_search_counter, self.grid_search_size * self.iter_number), end='\r')
        return step

    def print_stats(self, step, loss, correct, total):
        if step % 100 == 0:
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
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i >> j) & 1
                data[i, j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3 + case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description(
            "{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()


# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=9)
