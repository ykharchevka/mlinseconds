# There are 2 functions defined from input. One easy one and one hard one.
# On training data easy and hard functions produce same result and on
# test data you need to predict hard function.
# Easy function - depends on fewer inputs.
# Hard function - depends on more inputs.
# Easy and hard function depends on different inputs.
# Functions is a random functions of n-inputs, it's guarantee that
# functions depends on n inputs.
# For example:
# Inputs:
# x0, x1, x2, x3, x4, x5, x6, x7
# Easy function:
# x0^x1
# Hard function:
# x2^x3^x4^x5^x6^x7
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from ..utils import solutionmanager as sm

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        # sm.SolutionManager.print_hint("Hint[1]: NN usually learn easiest function, you need to learn hard one")
        self.solution = solution

        nn_width_list = [input_size] + [solution.nn_width] * solution.nn_depth + [output_size]
        self.layers = nn.ModuleList(nn.Linear(nn_width_list[i], nn_width_list[i + 1]) for i in range(len(nn_width_list) - 1))
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(i, affine=False, track_running_stats=False) for i in nn_width_list[1:])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = self.batch_norms[i](x)
                x = self.solution.activations[self.solution.activations_hidden](x)
        x = self.solution.activations[self.solution.activations_output](x)
        return x

    def calc_loss(self, output, target):
        bce_loss = nn.BCELoss()
        return bce_loss(output, target)

    def calc_predict(self, output):
        return output.round()

class Solution():
    def __init__(self):
        self = self
        self.activations = {
            'sg': nn.Sigmoid(),
            'th': nn.Tanh(),
            'r0': nn.ReLU(),
            'rr': nn.RReLU(0.1, 0.3),
            'r6': nn.ReLU6(),
            'pr': nn.PReLU(),
            'el': nn.ELU(),  # Exponential Linear Unit
            'sl': nn.SELU(),  # Self-Normalizing Neural Networks
            'yr': nn.LeakyReLU(0.1)
        }
        self.activations_hidden = 'r0'
        self.activations_output = 'sg'
        self.nn_depth = 3
        self.nn_width = 32
        self.learning_rate = 0.1
        self.momentum = 0.9

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        bce_loss = nn.BCELoss()
        train_total = train_data.size()[0]
        train_data_easy = torch.Tensor()
        train_target_easy = torch.Tensor()
        train_data_hard = torch.Tensor()
        train_target_hard = torch.Tensor()
        test_data = context.case_data.test_data[0]
        test_target = context.case_data.test_data[1]

        easy_model = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        easy_model.train()
        easy_optimizer = optim.SGD(easy_model.parameters(), lr=0.1, momentum=0.9)
        while True:
            easy_optimizer.zero_grad()
            output = easy_model.forward(train_data)
            predict = output.round()
            correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            if correct / train_total >= 0.75:
                print('Done training easy model')
                easies = []
                hards = []
                for k, v in enumerate(zip(train_data, predict, train_target)):
                    if v[1] != v[2]:
                        hards.append(k)
                    else:
                        easies.append(k)
                train_data_easy = train_data[easies, :]
                train_target_easy = train_target[easies, :]
                train_data_hard = train_data[hards, :]
                train_target_hard = train_target[hards, :]
                break
            loss = bce_loss(output, train_target)
            loss.backward()
            easy_optimizer.step()

        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        while True:
            optimizer.zero_grad()
            train_output = model.forward(train_data_easy)
            train_predict = train_output.round()
            train_correct = train_predict.eq(train_target_easy.view_as(train_predict)).long().sum().item()
            if train_correct / train_data_easy.size()[0] == 1.:
                print('Done training core model on easy subset')
                break
            train_loss = bce_loss(train_output, train_target_easy)
            train_loss.backward()
            optimizer.step()

        step = train_correct = 0
        while True:
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1:
                break
            optimizer.zero_grad()
            train_output = model.forward(train_data_hard)
            train_predict = train_output.round()
            train_correct = train_predict.eq(train_target_hard.view_as(train_predict)).long().sum().item()
            train_loss = bce_loss(train_output, train_target_hard)

            test_output = model.forward(test_data)
            test_predict = test_output.round()
            test_correct = test_predict.eq(test_target.view_as(test_predict)).long().sum().item()
            test_loss = bce_loss(test_output, test_target)

            if step < 50:
                print('{:>3}. Train_hard: {:>3}/{:>3}={:>8} < {:>8}; Test: {:>3}/{:>3}={:>8} < {:>8}'.format(
                    step,
                    train_correct, train_data_hard.size()[0], round(train_correct / train_data_hard.size()[0], 5), round(train_loss.item(), 5),
                    test_correct, test_data.size()[0], round(test_correct / test_data.size()[0], 5), round(test_loss.item(), 5))
                )
            train_loss.backward()
            optimizer.step()
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
        self.size_limit = 1000000
        self.test_limit = 0.75

class DataProvider:
    def __init__(self):
        self.number_of_cases = 20

    def full_func(self, input_size):
        while True:
            table = torch.ByteTensor(1<<input_size).random_(0, 2)
            vals = torch.ByteTensor(input_size, 2).zero_()
            depend_count = 0
            for i in range(input_size):
                for ind in range(1<<input_size):
                    if table[ind].item() != table[ind^(1<<i)].item():
                        depend_count += 1
                        break
            if depend_count == input_size:
                return table

    def tensor_to_int(self, tensor):
        tensor = tensor.view(-1)
        res = 0
        for x in tensor:
            res = (res<<1)+x.item()
        return res

    def int_to_tensor(self, ind, tensor):
        for i in range(tensor.size(0)):
            tensor[i] = (ind >> i)&1

    def create_data(self, seed, easy_table, hard_table, easy_input_size, hard_input_size, easy_correct):
        input_size = easy_input_size + hard_input_size
        data_size = 1 << input_size
        data = torch.ByteTensor(data_size, input_size)
        target = torch.ByteTensor(data_size, 1)
        count = 0
        for ind in range(data_size):
            self.int_to_tensor(ind, data[count])
            easy_ind = ind & ((1 << easy_input_size)-1)
            hard_ind = ind >> easy_input_size
            easy_value = easy_table[easy_ind].item()
            hard_value = hard_table[hard_ind].item()
            target[count, 0] = hard_value
            if not easy_correct or easy_value == hard_value:
                count += 1
        data = data[:count,:]
        target = target[:count,:]
        perm = torch.randperm(count)
        data = data[perm]
        target = target[perm]
        return (data.float(), target.float())

    def create_case_data(self, case):
        easy_input_size = 2
        hard_input_size = 6

        random.seed(case)
        torch.manual_seed(case)
        easy_table = self.full_func(easy_input_size)
        hard_table = self.full_func(hard_input_size)
        train_data, train_target = self.create_data(case, easy_table, hard_table, easy_input_size, hard_input_size, True)
        test_data, test_target = self.create_data(case, easy_table, hard_table, easy_input_size, hard_input_size, False)
        perm = torch.randperm(train_data.size(1))
        train_data = train_data[:,perm]
        test_data = test_data[:,perm]
        return sm.CaseData(case, Limits(), (train_data, train_target), (test_data, test_target)).set_description("Easy {} inputs and hard {} inputs".format(easy_input_size, hard_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=1)
