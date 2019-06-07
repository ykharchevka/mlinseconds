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
from ..utils.gridsearch import GridSearch


class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        # sm.SolutionManager.print_hint("Hint[1]: NN usually learn easiest function, you need to learn hard one")
        self.solution = solution

        if solution.grid_search.enabled:
            assert solution.iter >= 0
            torch.manual_seed(solution.iter)

        nn_width_list = [input_size] + [solution.nn_width_main] * solution.nn_depth_main + [output_size]
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
        self.sols = {}
        self.stats = {}
        self.predictions = {}
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
        self.nn_width_easy = 128
        self.nn_width_easy_grid = [128]
        self.learning_rate_easy = 0.1
        self.learning_rate_easy_grid = [0.1]
        self.momentum_easy = 0.9
        self.momentum_easy_grid = [0.9]
        self.nn_depth_main = 3
        self.nn_depth_main_grid = [3]
        self.nn_width_main = 32
        self.nn_width_main_grid = [32]
        self.learning_rate_main = 0.5
        self.learning_rate_main_grid = [0.5]
        self.momentum_main = 0.9
        self.momentum_main_grid = [0.9]
        self.iter = -1
        self.iter_number = 20
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(False)
        self.grid_search_counter = 0
        self.grid_search_size = eval(str(self.iter_number) + '*' + '*'.join([str(len(v)) for k, v in self.__dict__.items() if k.endswith('_grid')]))

    def __del__(self):
        for item in sorted(self.stats.items(), key=lambda kv: kv[1][0], reverse=True):
            print(item[1][1])

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    def get_key(self):
        return "eW{:02d}_eL{:.8f}_eM{:02f}_mD{:02d}_mW{:02d}_mL{:.8f}_mM{:.2f}".format(
            self.nn_width_easy, self.learning_rate_easy, self.momentum_easy, self.nn_depth_main, self.nn_width_main, self.learning_rate_main, self.momentum_main)

    def save_experiment_summary(self, key, prediction):
        if not key in self.sols:
            self.sols[key] = 0
            self.predictions[key] = []
        self.sols[key] += 1
        self.predictions[key].append(prediction)
        if self.sols[key] == self.iter_number:
            self.stats[key] = (
                np.sum(prediction), '{}: predicted = {:.8f}+-{:.8f}'.format(
                    key, np.mean(self.predictions[key]), np.std(self.predictions[key])))

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        key = self.get_key()
        if key in self.sols and self.sols[key] == -1:
            return
        bce_loss = nn.BCELoss()
        test_data = context.case_data.test_data[0]
        test_target = context.case_data.test_data[1]
        test_total = test_data.size()[0]

        easy_model = nn.Sequential(
            nn.Linear(8, self.nn_width_easy),
            nn.BatchNorm1d(self.nn_width_easy, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(self.nn_width_easy, 1),
            nn.Sigmoid(),
        )

        data_batches = []
        for col in range(8):
            train_data_variation = train_data.clone()
            train_data_variation[:, col] = train_data_variation[:, col].add(-1).abs()
            data_batches.append(train_data_variation)
        easy_model.train()
        easy_optimizer = optim.SGD(easy_model.parameters(), lr=self.learning_rate_easy, momentum=self.momentum_easy)
        easy_cols_search = [0] * 8
        easy_step = 0
        while True:
            easy_optimizer.zero_grad()
            easy_output = easy_model.forward(data_batches[easy_step % 8])
            easy_predict = easy_output.round()
            easy_correct = easy_predict.eq(train_target.view_as(easy_predict)).long().sum().item()
            easy_loss = bce_loss(easy_output, train_target)
            easy_loss.backward()
            easy_optimizer.step()
            easy_cols_search[easy_step % 8] = easy_correct
            easy_step += 1
            if easy_step > 32:
                break

        train_data_hard = train_data.clone()
        train_data_hard[:, [i[0] for i in sorted(enumerate(easy_cols_search), key=lambda v: v[1])[:2]]] = 0.

        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate_main, momentum=self.momentum_main)
        step = 0
        test_predict_ratio = 0.
        while True:
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1:
                break
            optimizer.zero_grad()
            train_output = model.forward(train_data_hard)
            train_predict = train_output.round()
            train_correct = train_predict.eq(train_target.view_as(train_predict)).long().sum().item()
            train_loss = bce_loss(train_output, train_target)

            test_output = model.forward(test_data)
            test_predict = test_output.round()
            test_correct = test_predict.eq(test_target.view_as(test_predict)).long().sum().item()
            test_loss = bce_loss(test_output, test_target)
            test_predict_ratio = test_correct / test_total
            if test_predict_ratio >= 0.75:
                break

            if step % 200 == 0 and False:
                print('{:>4}. Train_hard: {:>3}/{:>3}={:>8} < {:>8}; Test: {:>3}/{:>3}={:>8} < {:>8}'.format(
                    step,
                    train_correct, train_data_hard.size()[0], round(train_correct / train_data_hard.size()[0], 5), round(train_loss.item(), 5),
                    test_correct, test_data.size()[0], round(test_correct / test_data.size()[0], 5), round(test_loss.item(), 5))
                )
            train_loss.backward()
            optimizer.step()
            step += 1
        if self.grid_search.enabled:
            self.save_experiment_summary(key, test_predict_ratio)
            self.grid_search_counter += 1
            print('{:>8} / {:>8}'.format(self.grid_search_counter, self.grid_search_size), end='\r')
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
sm.SolutionManager(Config()).run(case_number=-1)
