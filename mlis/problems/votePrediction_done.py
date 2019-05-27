# There are random function from 8 inputs.
# There are random input vector of size 8 * number of voters.
# We calculate function number of voters times and sum result.
# We return 1 if sum > voters/2, 0 otherwise
# We split data in 2 parts, on first part you will train and on second
# part we will test
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils.gridsearch import GridSearch

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.solution = solution

        # different seed for removing noise
        if solution.grid_search.enabled:
            torch.manual_seed(solution.random)

        self.voters_count = input_size // 8
        width_voters = [8] + [solution.nn_width_voters] * solution.nn_depth_voters + [output_size]
        self.layers_voters = nn.ModuleList(nn.Linear(width_voters[i], width_voters[i + 1]) for i in range(len(width_voters) - 1))
        self.batch_norms_voters = nn.ModuleList(nn.BatchNorm1d(i, affine=False, track_running_stats=False) for i in width_voters[1:])
        width_main = [self.voters_count] + [solution.nn_width_main] * solution.nn_depth_main + [1]
        self.layers_main = nn.ModuleList(nn.Linear(width_main[i], width_main[i + 1]) for i in range(len(width_main) - 1))
        self.batch_norms_main = nn.ModuleList(nn.BatchNorm1d(i, affine=False, track_running_stats=False) for i in width_main[1:])

    def forward(self, x):
        x = x.view(x.size(0) * self.voters_count, -1)
        for i, layer in enumerate(self.layers_voters):
            x = layer(x)
            if i + 1 < len(self.layers_voters):
                x = self.batch_norms_voters[i](x)
                x = self.solution.activations[self.solution.activations_hidden](x)
        x = self.solution.activations[self.solution.activations_output](x)
        x = x.view(x.size(0) // self.voters_count, -1)
        for i, layer in enumerate(self.layers_main):
            x = layer(x)
            if i + 1 < len(self.layers_main):
                x = self.batch_norms_main[i](x)
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
        self.advanced_is = 8
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
        self.nn_depth_voters = 2
        # self.nn_depth_voters_grid = [2]
        self.nn_width_voters = 20
        # self.nn_width_voters_grid = [20]
        self.nn_depth_main = 1
        # self.nn_depth_main_grid = [1]
        self.nn_width_main = 40
        # self.nn_width_main_grid = [40]
        self.batch_size = 1024  # too small == lose speedup from vectorization; too big == too long per iteration for large datasets
        # self.batch_size_grid = [1024]
        self.learning_rate = 0.9
        self.learning_rate_grid = [0.9]
        self.momentum = 0.8
        self.momentum_grid = [0.8]
        self.learning_rate_advanced = 0.4274
        self.learning_rate_advanced_grid = [0.4274]
        self.momentum_advanced = 0.91
        self.momentum_advanced_grid = [0.91]
        self.weight_decay = 0.
        # self.weight_decay_grid = [0.]
        self.random = 1
        self.random_grid = [_ for _ in range(1, 11)]
        self.best_step = 1000
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(False)
        self.grid_search_counter = 0
        self.grid_search_size = eval('*'.join([str(len(v)) for k, v in self.__dict__.items() if k.endswith('_grid')]))

    def __del__(self):
        for item in sorted(self.stats.items(), key=lambda kv: kv[1][0]):
            print(item[1][1])

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    def get_key(self):
        return "{0:}{1:}_dv{2:02d}_wv{3:04d}_dm{4:02d}_wm{5:04d}_bs{6:03d}_lr{7:.6f}_mm{8:.6f}_la{9:.6f}_ma{10:.6f}".format(
            self.activations_hidden, self.activations_output,
            self.nn_depth_voters, self.nn_width_voters, self.nn_depth_main, self.nn_width_main,
            self.batch_size, self.learning_rate, self.momentum, self.learning_rate_advanced, self.momentum_advanced)

    def save_experiment_summary(self, key, correct, total, loss, time_left, step, is_test):
        if not key in self.sols:
            self.sols[key] = 0
            self.predictions[key] = []
            self.losses[key] = []
            self.time_lefts[key] = []
            self.steps_walks[key] = []
            self.is_tests[key] = []
        self.sols[key] += 1
        self.predictions[key].append(correct / total)
        self.losses[key].append(loss.item())
        self.time_lefts[key].append(time_left)
        self.steps_walks[key].append(step + 1)
        self.is_tests[key].append(int(is_test))
        if self.sols[key] == len(self.random_grid):
            self.stats[key] = (
                np.mean(self.losses[key]) + 10 * np.std(self.losses[key]), # criterion for sort
                '{}: worst pred-n = {:.8f}, loss = {:.8f}+-{:.8f}, worst time left = {:.4f}, worst steps = {:>4}, isTest ratio = {}'
                    .format(key, np.min(self.predictions[key]), np.mean(self.losses[key]), np.std(self.losses[key]),
                            np.min(self.time_lefts[key]), np.max(self.steps_walks[key]),
                            np.sum(self.is_tests[key]) / len(self.is_tests[key])
                            )
            )

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        key = self.get_key()
        if key in self.sols and self.sols[key] == -1:
            return
        step = 0
        # Put model in train mode
        model.train()
        batches = int(train_data.size(0) / self.batch_size)

        if model.voters_count < self.advanced_is:
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate_advanced, momentum=self.momentum_advanced)
        # optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate, momentum=self.momentum, alpha=0.99, weight_decay=self.weight_decay, eps=1e-08, centered=False)
        # optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        test_data = context.case_data.test_data[0]
        test_target = context.case_data.test_data[1]

        while True:
            ind = step % batches
            start_ind = self.batch_size * ind
            end_ind = start_ind + self.batch_size
            data = train_data[start_ind:end_ind]
            target = train_target[start_ind:end_ind]
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                output = model(data)
                predict = model.calc_predict(output)
                correct = predict.eq(target.view_as(predict)).long().sum().item()
                total = predict.view(-1).size(0)
                loss = model.calc_loss(output, target)
                self.save_experiment_summary(key, correct, total, loss, time_left, step, False)
                break
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
                model.eval()
                test_output = model(test_data)
                test_predict = model.calc_predict(test_output)
                test_correct = test_predict.eq(test_target.view_as(test_predict)).long().sum().item()
                test_total = test_target.view(-1).size(0)
                if test_correct == test_total:
                    loss = model.calc_loss(test_output, test_target)
                    self.save_experiment_summary(key, test_correct, test_total, loss, time_left, step, True)
                    break
            # calculate loss
            loss = model.calc_loss(output, target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            # self.print_stats(step, loss, correct, total)
            self.grid_search.log_step_value('loss', loss.item(), step)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        if self.grid_search.enabled:
            self.grid_search_counter += 1
            print('{:>8} / {:>8}'.format(self.grid_search_counter, self.grid_search_size), end='\r')
        return step

    def print_stats(self, key, time_left, step, loss, correct, total):
        if step % 10 == 0:
            print("{}: Time left = {:<4}, Step = {:>2}, Prediction = {}/{}, Loss = {}"
                  .format(key, round(time_left, 2), step, correct, total, loss.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def get_index(self, tensor_index):
        index = 0
        for i in range(tensor_index.size(0)):
            index = 2*index + tensor_index[i].item()
        return index

    def calc_value(self, input_data, function_table, input_size, input_count_size):
        count = 0
        for i in range(input_count_size):
            count += function_table[self.get_index(input_data[i*input_size: (i+1)*input_size])].item()
        if count > input_count_size/2:
            return 1
        else:
            return 0

    def create_data(self, data_size, input_size, input_count_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_table = torch.ByteTensor(function_size).random_(0, 2)
        total_input_size = input_size*input_count_size
        data = torch.ByteTensor(data_size, total_input_size).random_(0, 2)
        target = torch.ByteTensor(data_size)
        for i in range(data_size):
            target[i] = self.calc_value(data[i], function_table, input_size, input_count_size)
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        input_size = 8
        data_size = (1<<input_size)*32
        input_count_size = case

        data, target = self.create_data(2*data_size, input_size, input_count_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs per voter and {} voters".format(input_size, input_count_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
