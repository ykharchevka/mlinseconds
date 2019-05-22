# There are random function from 8 inputs.
# There are random input vector of size 8 * number of voters.
# We calculate function number of voters times and sum result.
# We return 1 if sum > voters/2, 0 otherwise
# We split data in 2 parts, on first part you will train and on second
# part we will test
"""
TODOs:
0. try adam while tuning lr -> beta=0.9+hidden_units+mini_batch_size -> layers+learning_rate_decay
0. look at laptop + try sgd with momentum
1. Add mini batches: https://github.com/romankoshlyak/mlinseconds/commit/c714cce1a90c2578e9da888157e70be3463b4b79?fbclid=IwAR1qgBh_VAZFsfxL7IYJkLLr4jVFE85R90lPre0floK_kw1_DYJ3I_cMZB4
   ( https://www.youtube.com/watch?v=4qJaSmvhxi8 ).
   - As far as I understood: we would better learn not on the whole train subset each step at once, but focusing at those parts of
     train subset, accuracy for which was low. Currently: training all minibatches iteratively until the worst L1 loss on every batch is < 0.3
2. Reducing network size in case of overfitting could help.
3. To pass all test cases only mini-batch gradient descent technic helped (especially with case 10).
   But it was fast enough only in combination with error estimation across mini-batches of training examples.
   Also it's important to use a proper batch size. Tried all recommended values like 32, 64, 128, 256.
   https://github.com/svoit/mlinseconds/commit/edc2dffa62ed74320c9d7c970e38c2764553b602?fbclid=IwAR1nu0frwNFeY9xCkkMeq6uxpHEF_yOzm3PI6peR1X-h15kAjxFT0T32i9U
4. Maybe add "x = (x - 0.5)*9" to forward def beginning. Hint: we are not obliged to normalize targets same way as we normalize inputs!
5. implement random search within the given range
6. use log scale for lr random search: r = -4 * np.random.rand(); alpha = 10**r; beta: similar as for alpha, but (1- beta)
7. cavier approach: run multiple instances on different pcs to speedup hparams search
8. https://www.youtube.com/watch?v=5qefnAek8OA&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=30
9. batch norm at test, softmax regression and classification; local optima
X. https://www.youtube.com/watch?v=QrzApibhohY&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=13
"""
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

        self.layers_size = [input_size] + [self.solution.hidden_size
                                           for _ in range(self.solution.layers_number)] + [output_size]
        self.linears = nn.ModuleList([nn.Linear(from_size, to_size)
                                      for (from_size, to_size) in zip(self.layers_size, self.layers_size[1:])])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(batch_size)
                                          for batch_size in self.layers_size[1:]])
        self.activations = [self.solution.activations[self.solution.activation_hidden]
                            for _ in self.layers_size[2:]] + [self.solution.activations[self.solution.activation_output]]

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.)

        self.linears.apply(init_weights)

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.batch_norms[i](x)
            x = self.activations[i](x)
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
        self.stats = {}
        self.worst_loss = {}
        self.worst_prediction = {}
        self.worst_time_left = {}
        self.worst_steps = {}
        self.best_step = 1000
        self.activations = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'rrelu0103': nn.RReLU(0.1, 0.3),
            'relu6': nn.ReLU6(),
            'prelu': nn.PReLU(),
            'elu': nn.ELU(),  # Exponential Linear Unit
            'selu': nn.SELU(),  # Self-Normalizing Neural Networks
            'leakyrelu01': nn.LeakyReLU(0.1)
        }
        self.layers_number = 2
        self.layers_number_grid = [2, 3, 4, 5]
        self.hidden_size = 101
        self.hidden_size_grid = [50, 100, 150, 200]
        self.batch_size = 128
        self.batch_size_grid = [64, 128, 256]
        self.batch_loss = 0.3
        self.batch_loss_grid = [0.3]
        self.learning_rate = 0.01
        self.learning_rate_grid = [0.01, 0.05, 0.1]
        self.momentum = 0.9  # 0.8
        self.momentum_grid = [0.9]
        self.weight_decay = 0.
        self.weight_decay_grid = [0.0]
        self.do_batch_norm = True
        # self.do_batch_norm_grid = [False, True]
        self.activation_hidden = 'relu'  # 'tanh'
        # self.activation_hidden_grid = ['relu']
        self.activation_output = 'sigmoid'
        # self.activation_output_grid = self.activations.keys()
        self.random = 0
        self.random_grid = [_ for _ in range(10)]
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(True)
        self.grid_search_counter = 0
        self.grid_search_size = eval('*'.join([str(len(v)) for k, v in self.__dict__.items() if k.endswith('_grid')]))

    def __del__(self):
        for item in sorted(self.stats.items(), key=lambda kv: kv[1][0]):
            print(item[1][1])

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    def get_key(self):
        return "hL{0:02d}_hs{1:04d}_bs{2:03d}_lr{3:.4f}_mm{4:.2f}".format(
            self.layers_number, self.hidden_size, self.batch_size, self.learning_rate, self.momentum)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        key = self.get_key()
        if key in self.sols and self.sols[key] == -1:
            return
        step = 0
        loss = None
        # Put model in train mode
        model.train()

        batches = int(train_data.size(0) / self.batch_size)
        goodCount = 0
        goodLimit = batches
        goodBreak = False

        # optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                  alpha=0.99, weight_decay=self.weight_decay, eps=1e-08, centered=False)
        while True:
            ind = step % batches
            start_ind = self.batch_size * ind
            end_ind = start_ind + self.batch_size
            data = train_data[start_ind:end_ind]
            target = train_target[start_ind:end_ind]

            time_left = context.get_timer().get_time_left()
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

            diff = (output.data - target.data).abs()
            if diff.max() < self.batch_loss:
                goodCount += 1
                if goodCount >= goodLimit:
                    goodBreak = True
            else:
                goodCount = 0

            if (
                    # correct == total or
                    time_left < 0.1 or
                    goodBreak
                    # or (self.grid_search.enabled and step > 2000)
            ):
                output = model(train_data)
                predict = model.calc_predict(output)
                correct = predict.eq(train_target.view_as(predict)).long().sum().item()
                total = predict.view(-1).size(0)
                loss = model.calc_loss(output, train_target)
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
                    self.stats[key] = (
                        self.worst_loss[key], '{}: Prediction = {:.4f}, Loss = {:.4f}, Time left = {:.4f}, Step = {}'
                            .format(key, self.worst_prediction[key], self.worst_loss[key], self.worst_time_left[key],
                                    self.worst_steps[key])
                    )
                break

            # calculate loss and log it
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
sm.SolutionManager(Config()).run(case_number=10)
