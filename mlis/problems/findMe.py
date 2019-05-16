'''
No 10 steps condition for the task
TODOs:
0. What are the fastest activations -> TBD, loss -> TBD and optim -> probably SGD?
1. + Binary_cross_entropy vs BCEloss -> same
2. + BatchNorm1d + track running stats = False
3. + relu vs relu6
4. + relu vs prelu
5. inputs normalization: [0, 1] ->> [-0.5, 0.5] or [-4.5, 4.5]
6. use model averaging to improve the accuracy
 - so instead of one [24, 24, 24] network I created an architecture
 of three different [24, 24, 24] chains of hidden layers
 combined with a [72, 1] layer on top
7. use bilinear layers to get feature crosses (not helped with generalCpu)
 - mistake was doing this as one of the last steps, not one of the first steps
 - could've saved a lot of grid search time
8. https://www.youtube.com/watch?v=BOCLq2gpcGU&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=8
'''
# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
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
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.solution = solution
        # different seed for removing noise
        if self.solution.grid_search.enabled:
            torch.manual_seed(solution.random)
        self.hidden_size = self.solution.hidden_size
        self.linears = nn.ModuleList([nn.Linear(
            self.input_size if i == 0 else self.hidden_size,
            self.hidden_size if i != self.solution.layers_number-1 else self.output_size
        ) for i in range(self.solution.layers_number)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(
            self.hidden_size if i != self.solution.layers_number-1 else self.output_size, track_running_stats=False
        ) for i in range(self.solution.layers_number)])

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.solution.do_batch_norm:
                x = self.batch_norms[i](x)
            act_function = self.solution.activation_output if i == len(self.linears)-1 else self.solution.activation_hidden
            x = self.solution.activations[act_function](x)
        return x

    def calc_loss(self, output, target):
        loss_fn = nn.BCELoss()
        try:
            loss = loss_fn(output, target)
        except RuntimeError:
            print('Runtime error at hidden depth: {}, hidden size: {}, activations: {}->{}, batch norm: {}, '
                  'lr: {}, momentum: {}, random: {}'.
                  format(self.solution.layers_number,
                         self.solution.hidden_size,
                         self.solution.activation_hidden,
                         self.solution.activation_output,
                         self.solution.do_batch_norm,
                         self.solution.learning_rate,
                         self.solution.momentum,
                         self.solution.random))
            return False
        return loss

    def calc_predict(self, output):
        predict = output.round()
        return predict

class Solution():
    def __init__(self):
        self = self
        self.sols = {}
        self.worst_loss = {}
        self.worst_prediction = {}
        self.worst_time_left = {}
        self.worst_steps = {}
        self.best_step = 1000
        self.activations = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'rrelu0103': nn.RReLU(0.1, 0.3),
            'relu6': nn.ReLU6(),
            'prelu': nn.PReLU(),
            'elu': nn.ELU(),  # Exponential Linear Unit
            'selu': nn.SELU(),  # Self-Normalizing Neural Networks
            'leakyrelu01': nn.LeakyReLU(0.1)
        }
        self.layers_number = 3
        self.layers_number_grid = [2, 3, 5, 6, 10]
        self.hidden_size = 50
        self.hidden_size_grid = [10, 20, 30, 50]
        self.do_batch_norm = True
        self.do_batch_norm_grid = [False, True]
        self.activation_hidden = 'relu'
        self.activation_hidden_grid = self.activations.keys()
        self.activation_output = 'sigmoid'
        # self.activation_output_grid = self.activations.keys()
        self.learning_rate = 0.003
        self.learning_rate_grid = [0.0001, 0.001, 0.01, 0.1]
        self.momentum = 0.  # 0.8
        # self.momentum_grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.random = 0
        self.random_grid = [_ for _ in range(10)]
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(True)
        self.grid_search_counter = 0
        self.grid_search_size = eval('*'.join([str(len(v)) for k, v in self.__dict__.items() if k.endswith('_grid')]))

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    def get_key(self):
        return "{}_{}_{}_{}_{}_{}_{}".format(self.learning_rate, self.momentum, self.hidden_size, self.activation_hidden,
                                             self.activation_output, self.do_batch_norm, "{0:03d}".format(self.layers_number))

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        key = self.get_key()
        if key in self.sols and self.sols[key] == -1:
            return
        step = 0
        loss = None
        # Put model in train mode
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        # optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate, momentum=self.momentum,
        #                           alpha=0.99, weight_decay=0, eps=1e-08, centered=False)
        while True:
            time_left = context.get_timer().get_time_left()
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
            if correct == total or time_left < 0.1 or (self.grid_search.enabled and step > 100):
                if not key in self.sols:
                    self.sols[key] = 0
                    self.worst_prediction[key] = correct / total
                    self.worst_loss[key] = loss.item()
                    self.worst_time_left[key] = time_left
                    self.worst_steps[key] = step + 1
                self.sols[key] += 1
                self.worst_prediction[key] = min(self.worst_prediction[key], correct / total)
                self.worst_loss[key] = max(self.worst_loss[key], loss.item())
                self.worst_time_left[key] = min(self.worst_time_left[key], time_left)
                self.worst_steps[key] = max(self.worst_steps[key], step + 1)
                if self.sols[key] == len(self.random_grid):
                    print("{}: Prediction = {:.4f}, Loss = {:.4f}, Time left = {:.4f}, Step = {}".format(
                        key, self.worst_prediction[key], self.worst_loss[key], self.worst_time_left[key], self.worst_steps[key]))
                break
            # calculate loss
            loss = model.calc_loss(output, target)
            if not loss:
                break
            self.grid_search.log_step_value('loss', loss.item(), step)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            # self.print_stats(key, time_left, step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
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
        self.size_limit = 1_000_000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size  # 256
        function_input = torch.ByteTensor(function_size, input_size)
        for i in range(function_input.size(0)):
            fun_ind = i
            for j in range(function_input.size(1)):
                input_bit = fun_ind&1
                fun_ind = fun_ind >> 1
                function_input[i][j] = input_bit
        function_output = torch.ByteTensor(function_size).random_(0, 2)

        if data_size % function_size != 0:
            raise "Data gen error"

        data_input = torch.ByteTensor(data_size, input_size).view(-1, function_size, input_size)
        target = torch.ByteTensor(data_size).view(-1, function_size)
        for i in range(data_input.size(0)):
            data_input[i] = function_input
            target[i] = function_output
        data_input = data_input.view(data_size, input_size)
        target = target.view(data_size)
        if random_input_size > 0:
            data_random = torch.ByteTensor(data_size, random_input_size).random_(0, 2)
            data = torch.cat([data_input, data_random], dim=1)
        else:
            data = data_input
        perm = torch.randperm(data.size(1))
        data = data[:,perm]
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        data_size = 256*32  # 8_192
        input_size = 8
        random_input_size = min(32, (case-1)*4)  # 0...32 as case is up to 10

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(
            case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])
        ).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10_000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=10)
