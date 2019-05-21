# There are random function from 8 inputs.
# There are random input vector of size 8 * number of voters.
# We calculate function number of voters times and sum result.
# We return 1 if sum > voters/2, 0 otherwise
# We split data in 2 parts, on first part you will train and on second
# part we will test
"""
TODOs:
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
5. https://www.youtube.com/watch?v=QrzApibhohY&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=13

hL02_hs080_bs512_lr0.0010_mm0.90: Prediction = 1.0000, Loss = 0.0811, Time left = 0.2096, Step = 443
hL02_hs064_bs512_lr0.0010_mm0.90: Prediction = 0.9766, Loss = 0.1051, Time left = 0.0974, Step = 578
hL02_hs064_bs1024_lr0.0010_mm0.90: Prediction = 1.0000, Loss = 0.1084, Time left = 0.6469, Step = 282
hL02_hs160_bs512_lr0.0010_mm0.90: Prediction = 1.0000, Loss = 0.1124, Time left = 0.1853, Step = 311
hL02_hs080_bs1024_lr0.0010_mm0.90: Prediction = 1.0000, Loss = 0.1222, Time left = 0.8184, Step = 214
hL02_hs160_bs256_lr0.0010_mm0.90: Prediction = 0.9648, Loss = 0.1314, Time left = 0.0951, Step = 468
hL02_hs064_bs512_lr0.0005_mm0.90: Prediction = 0.9824, Loss = 0.1326, Time left = 0.0972, Step = 574
hL02_hs160_bs1024_lr0.0010_mm0.90: Prediction = 1.0000, Loss = 0.1580, Time left = 0.3062, Step = 161
hL02_hs080_bs512_lr0.0005_mm0.90: Prediction = 0.9902, Loss = 0.1591, Time left = 0.0973, Step = 524
hL02_hs064_bs256_lr0.0010_mm0.90: Prediction = 0.9414, Loss = 0.1625, Time left = 0.0965, Step = 674
hL02_hs080_bs256_lr0.0010_mm0.90: Prediction = 0.9414, Loss = 0.1632, Time left = 0.0968, Step = 607
hL02_hs064_bs1024_lr0.0005_mm0.90: Prediction = 1.0000, Loss = 0.1723, Time left = 0.3141, Step = 333
hL02_hs160_bs256_lr0.0005_mm0.90: Prediction = 0.9805, Loss = 0.1847, Time left = 0.0957, Step = 455
hL02_hs080_bs256_lr0.0005_mm0.90: Prediction = 0.9688, Loss = 0.1906, Time left = 0.0963, Step = 569
hL02_hs064_bs256_lr0.0005_mm0.90: Prediction = 0.9258, Loss = 0.2007, Time left = 0.0972, Step = 640
hL02_hs160_bs512_lr0.0005_mm0.90: Prediction = 0.9922, Loss = 0.2103, Time left = 0.0937, Step = 322
hL02_hs080_bs1024_lr0.0005_mm0.90: Prediction = 1.0000, Loss = 0.2206, Time left = 0.4514, Step = 257
hL02_hs320_bs512_lr0.0010_mm0.90: Prediction = 0.9316, Loss = 0.2318, Time left = 0.0888, Step = 185
hL02_hs160_bs1024_lr0.0005_mm0.90: Prediction = 1.0000, Loss = 0.2550, Time left = 0.2254, Step = 165
hL02_hs320_bs256_lr0.0010_mm0.90: Prediction = 0.8867, Loss = 0.3002, Time left = 0.0924, Step = 292
hL02_hs320_bs512_lr0.0005_mm0.90: Prediction = 0.9219, Loss = 0.3085, Time left = 0.0909, Step = 183
hL02_hs320_bs1024_lr0.0010_mm0.90: Prediction = 0.8740, Loss = 0.3509, Time left = 0.0768, Step = 88
hL02_hs320_bs256_lr0.0005_mm0.90: Prediction = 0.8711, Loss = 0.3651, Time left = 0.0930, Step = 288
hL02_hs320_bs1024_lr0.0005_mm0.90: Prediction = 0.9141, Loss = 0.3665, Time left = 0.0777, Step = 106
hL02_hs080_bs512_lr0.0001_mm0.90: Prediction = 0.9785, Loss = 0.3890, Time left = 0.0964, Step = 508
hL02_hs064_bs512_lr0.0001_mm0.90: Prediction = 0.9395, Loss = 0.3945, Time left = 0.0965, Step = 584
hL02_hs064_bs256_lr0.0001_mm0.90: Prediction = 0.8711, Loss = 0.4119, Time left = 0.0957, Step = 683
hL02_hs080_bs256_lr0.0001_mm0.90: Prediction = 0.9102, Loss = 0.4233, Time left = 0.0944, Step = 568
hL02_hs064_bs1024_lr0.0001_mm0.90: Prediction = 0.9482, Loss = 0.4325, Time left = 0.0951, Step = 398
hL02_hs080_bs1024_lr0.0001_mm0.90: Prediction = 0.9766, Loss = 0.4492, Time left = 0.0950, Step = 333
hL02_hs160_bs512_lr0.0001_mm0.90: Prediction = 1.0000, Loss = 0.4501, Time left = 0.2610, Step = 291
hL02_hs160_bs256_lr0.0001_mm0.90: Prediction = 1.0000, Loss = 0.4550, Time left = 0.3556, Step = 377
hL02_hs320_bs256_lr0.0001_mm0.90: Prediction = 0.9297, Loss = 0.5113, Time left = 0.0948, Step = 278
hL02_hs160_bs1024_lr0.0001_mm0.90: Prediction = 1.0000, Loss = 0.5139, Time left = 0.5265, Step = 145
hL02_hs320_bs512_lr0.0001_mm0.90: Prediction = 0.9531, Loss = 0.5399, Time left = 0.0889, Step = 195
hL02_hs640_bs1024_lr0.0005_mm0.90: Prediction = 0.7559, Loss = 0.5699, Time left = 0.0457, Step = 38
hL02_hs640_bs512_lr0.0005_mm0.90: Prediction = 0.7500, Loss = 0.5712, Time left = 0.0800, Step = 70
hL02_hs640_bs1024_lr0.0010_mm0.90: Prediction = 0.6973, Loss = 0.5781, Time left = 0.0500, Step = 41
hL02_hs640_bs256_lr0.0005_mm0.90: Prediction = 0.6836, Loss = 0.5830, Time left = 0.0870, Step = 132
hL02_hs320_bs1024_lr0.0001_mm0.90: Prediction = 0.9287, Loss = 0.5908, Time left = 0.0860, Step = 88
hL02_hs640_bs256_lr0.0001_mm0.90: Prediction = 0.7656, Loss = 0.5949, Time left = 0.0873, Step = 140
hL02_hs640_bs512_lr0.0010_mm0.90: Prediction = 0.7031, Loss = 0.5978, Time left = 0.0751, Step = 88
hL02_hs640_bs512_lr0.0001_mm0.90: Prediction = 0.7949, Loss = 0.6109, Time left = 0.0678, Step = 77
hL02_hs640_bs256_lr0.0010_mm0.90: Prediction = 0.6680, Loss = 0.6246, Time left = 0.0845, Step = 143
hL02_hs640_bs1024_lr0.0001_mm0.90: Prediction = 0.8271, Loss = 0.6326, Time left = 0.0468, Step = 39
>>>>>>> WiP on votePrediction
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
        self.layers_number_grid = [2]
        self.hidden_size = 80
        self.hidden_size_grid = [64, 80, 160, 320, 640]
        self.batch_size = 256
        self.batch_size_grid = [256, 512, 1024]
        self.learning_rate = 0.001
        self.learning_rate_grid = [0.0001, 0.0005, 0.001]
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
        return "hL{0:02d}_hs{1:03d}_bs{2:03d}_lr{3:.4f}_mm{4:.2f}".format(
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
            if diff.max() < 0.3:
                goodCount += 1
                if goodCount >= goodLimit:
                    goodBreak = True
            else:
                goodCount = 0

            if (
                    correct == total or
                    time_left < 0.1 or
                    goodBreak
                    # or (self.grid_search.enabled and step > 2000)
            ):
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
