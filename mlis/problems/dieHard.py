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

# TODOs: 1. add random search instead of grid search; 2. think to apply PPO as another case of search as well

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
        # different seed for removing noise
        if solution.grid_search.enabled:
            torch.manual_seed(solution.random)

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
        loss = bce_loss(output, target)
        return loss

    def calc_predict(self, output):
        predict = output.round()
        return predict

class Solution():
    def __init__(self):
        self = self
        self.sols = {}
        self.stats = {}
        self.losses = {}
        self.predictions = {}
        self.time_lefts = {}
        self.steps_walks = {}
        self.is_tests = {}
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
        # self.activations_hidden_grid = ['r0']
        self.activations_output = 'sg'
        # self.activations_output_grid = ['sg']

        # TODO: consider varying hidden neurons count from layer to layer

        depth_width_pairs = ((1, 99999), (2, 994), (3, 704), (4, 575), (5, 498), (6, 445), (7, 406), (8, 376), (9, 352), (10, 332), (390, 50))
        depth_width_values = []
        for layers, neurons in depth_width_pairs:
            depth_width_values.extend([(layers, i) for i in range(8, neurons + 1)])

        self.nn_dim_id = 8
        self.nn_dim_id_grid = np.random.randint(0, len(depth_width_values), 20)
        self.nn_depth = depth_width_values[self.nn_dim_id][0]
        self.nn_width = depth_width_values[self.nn_dim_id][1]
        self.learning_rate = 0.1
        self.learning_rate_grid = 10 ** np.random.uniform(np.log10(1e-5), np.log10(1e2), 30)
        self.momentum = 0.9
        self.momentum_grid = np.random.uniform(0.0001, 0.9999, 10)
        self.early_stop = 0.75
        self.early_stop_grid = np.random.uniform(0.75, 1.0, 5)
        self.random = 1
        self.random_grid = [1]
        # self.random_grid = [_ for _ in range(1, 21)]
        self.best_step = 1000
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(True)
        self.grid_search_counter = 0
        self.grid_search_size = eval('*'.join([str(len(v)) for k, v in self.__dict__.items() if k.endswith('_grid')]))

    def __del__(self):
        if self.grid_search.enabled:
            print('----------------------------------------')
            for item in sorted(self.stats.items(), key=lambda kv: kv[1][0], reverse=True):
                print(item[1][1])

    def get_key(self):
        return "d{0:02d}_w{1:04d}_lr{2}{3:.10f}_mm{4:.10f}_es{5:.10f}".format(
            self.nn_depth, self.nn_width, '0' if self.learning_rate < 10 else '',
            self.learning_rate, self.momentum, self.early_stop)

    def save_experiment_summary(self, key, train_correct, train_total, train_loss, test_correct, test_total, test_loss,
                                time_left, step, is_test):
        if not key in self.sols:
            self.sols[key] = 0
            self.predictions[key] = []
            self.losses[key] = []
            self.time_lefts[key] = []
            self.steps_walks[key] = []
            self.is_tests[key] = []
        self.sols[key] += 1
        self.predictions[key].append(test_correct / test_total)
        self.losses[key].append(test_loss.item())
        self.time_lefts[key].append(time_left)
        self.steps_walks[key].append(step + 1)
        self.is_tests[key].append(int(is_test))
        if self.sols[key] == len(self.random_grid):
            self.stats[key] = (
                np.mean(self.predictions[key]),
                '{}: t.pred-n = {:2.8f}, t.loss = {:.8f}+-{:.8f}, worst time left = {:.4f}, worst steps = {:>4}, isTest ratio = {}'.
                    format(key, np.min(self.predictions[key]), np.mean(self.losses[key]), np.std(self.losses[key]),
                           np.min(self.time_lefts[key]), np.max(self.steps_walks[key]),
                           np.sum(self.is_tests[key]) / len(self.is_tests[key]))
            )
            print('{:.8f} == {}'.format(self.stats[key][0], self.stats[key][1]))
            # self.stats[key] = (
            #     np.mean(self.losses[key]) + 10 * np.std(self.losses[key]), # criterion for sort
            #     '{}: worst pred-n = {:.8f}, loss = {:.8f}+-{:.8f}, worst time left = {:.4f}, worst steps = {:>4}, isTest ratio = {}'
            #         .format(key, np.min(self.predictions[key]), np.mean(self.losses[key]), np.std(self.losses[key]),
            #                 np.min(self.time_lefts[key]), np.max(self.steps_walks[key]),
            #                 np.sum(self.is_tests[key]) / len(self.is_tests[key])
            #                 )
            # )

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        test_data = context.case_data.test_data[0]
        test_target = context.case_data.test_data[1]

        key = self.get_key()
        if key in self.sols and self.sols[key] == -1:
            return

        # Put model in train mode
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        step = train_correct = train_total = test_correct = test_total = 0
        train_loss = test_loss = 999.
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left or training done, stop training
            if time_left < 0.1 or (train_correct >= train_total * self.early_stop and train_total > 0):
                if self.grid_search.enabled:
                    self.save_experiment_summary(key, train_correct, train_total, train_loss,
                                                 test_correct, test_total, test_loss, time_left, step, is_test=False)
                break
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            train_output = model(train_data)
            # if x < 0.5 predict 0 else predict 1
            train_predict = model.calc_predict(train_output)
            # Number of correct predictions
            train_correct = train_predict.eq(train_target.view_as(train_predict)).long().sum().item()
            # Total number of needed predictions
            train_total = train_predict.view(-1).size(0)
            # calculate loss
            train_loss = model.calc_loss(train_output, train_target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            train_loss.backward()
            # print progress of the learning
            if step % 1 == 0:
                model.eval()
                test_output = model(test_data)
                test_predict = model.calc_predict(test_output)
                test_correct = test_predict.eq(test_target.view_as(test_predict)).long().sum().item()
                test_total = test_target.view(-1).size(0)
                test_loss = model.calc_loss(test_output, test_target)
                if not self.grid_search.enabled and step < 50:
                    self.print_stats(step, train_correct, train_total, train_loss, test_correct, test_total, test_loss)
                if test_correct == test_total:
                    self.save_experiment_summary(key, train_correct, train_total, train_loss,
                                                 test_correct, test_total, test_loss, time_left, step, is_test=True)
                    break
                model.train()
            # self.grid_search.log_step_value('loss', loss.item(), step)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1

        if self.grid_search.enabled:
            self.grid_search_counter += 1
            print('{:>8} / {:>8}'.format(self.grid_search_counter, self.grid_search_size), end='\r')

        return step

    def print_stats(self, step, train_correct, train_total, train_loss, test_correct, test_total, test_loss):
        print("Step = {:>5} Train: {:>3}/{:>3} = {:.4f}, error = {:.8f}; Test = {:>3}/{:>3} = {:.4f}, error = {:.8f}".
              format(step, train_correct, train_total, train_correct / train_total, train_loss.item(),
                     test_correct, test_total, test_correct / test_total, test_loss.item()))

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
