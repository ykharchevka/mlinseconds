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

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.)

        self.linears.apply(init_weights)

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


"""
Work log:
1. Walked through the dataset generation trying to explain the logic behind it to myself
2. Applied all solutions known from generalCpu.
3. Applied noise reduction provided by Roman and tuned a bit:
   + gathering worst prediction, loss, time left and steps among 10 random launches for each hparams combination.
   + printing them out sorted by BCE loss descending.
   + added a fancy progress counter that prints current experiment id and overall experiments quantity of grid search being run.
4. Launched grid search varying
       activation functions (9 presets) * batch norm (on / off) * randomization while * couple of learning rates
   keeping layers amount = 3 and neurons amount = 50 ->  batch norm = True was always better, relu showed itself better.
5. + normalize data 0...1 -> -1...1 (didn't help, however is a good practice as simplifies the objective function and allows usning smaller learning rates + learn with less steps)
   - in practice degraded prediction by ~15%; decreasing lr = even more loss, increasing lr = negative loss %)
6. + initialize weights: tanh -> xavier (helped a bit: -0.1 loss) as Andrew Ng recommended
7. achieved 100% accuracy on case 10 within 80s for train set, while test set gives 75% accuracy
   - overfitting, high variance (bias looks ok as train set gives 100% accuracy and just dev=test set is the problem)
   + high bias     (train=high_loss(15%)+test=high_loss(16%) or train=high_loss(15%)+test=high_loss(30%)):
        + bigger network (data not fitting at all)
        + train longer (data not fitting well)
        + (NN architecture search)
   + high variance (train=low_loss(1%)+test=high_loss(11%)   or train=high_loss(15%)+test=high_loss(30%)):
        + more data
        + regularization approaches:
            + batch norm (done)
            + early stopping (done)
            + l2 regularization (add penalty for weight sizes) -> use weight decay param in optim
                (?) 0.0 looks to be the best... 0.2 -> Accuracy_0.56_0.54 in 1754steps, 0.8 -> 0.52_0.52 in 1323steps;
                    0.001_0.0_50_tanh_sigmoid_True_005_0.0: Prediction = 0.8149, Loss = 0.5425, Time left = 71.6493, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.01: Prediction = 0.7545, Loss = 0.5867, Time left = 71.5825, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.015: Prediction = 0.6868, Loss = 0.6193, Time left = 71.2361, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.2: Prediction = 0.5510, Loss = 0.6810, Time left = 70.2195, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.8: Prediction = 0.5378, Loss = 0.6893, Time left = 71.6796, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.3: Prediction = 0.5266, Loss = 0.6895, Time left = 71.3551, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.99: Prediction = 0.5502, Loss = 0.6901, Time left = 71.0996, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.6: Prediction = 0.5428, Loss = 0.6904, Time left = 66.3860, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.1: Prediction = 0.5739, Loss = 0.6932, Time left = 71.6135, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.7: Prediction = 0.4995, Loss = 0.6954, Time left = 71.3379, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.9: Prediction = 0.5323, Loss = 0.6971, Time left = 67.8226, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_1.0: Prediction = 0.5385, Loss = 0.7009, Time left = 71.5290, Step = 202
                    0.001_0.0_50_tanh_sigmoid_True_005_0.5: Prediction = 0.5381, Loss = 0.7143, Time left = 70.7954, Step = 202
            + dropout
            + data augmentation
        + (NN architecture search)
   + low bias and low variance: train=low_loss(0.5%)+test=low_loss(1%)
8. Launched grid search varying ANN layers quantity (8 cases), neurons quantity (11 cases) and learning rate (8 cases)
   = 7040 cases limiting steps to 200 and keeping 2 seconds termination criterion on -> found 4 cases giving 100% accuracy
   on training subset.
9. Tried all 4 cases:
    neurons=32, layers=3, lr=0.05 -> loss=0.03&0.04 with test correct=8177/8192
    neurons=32, layers=3, lr=0.01 -> loss=0.12&0.12 with test correct=8189/8192
    Strange aspect: parameters with worse BCELoss gives better prediction
    neurons=32, layers=5, lr=0.01 -> loss=0.15&0.16 with test correct=8175/8192
    neurons=16, layers=5, lr=0.01 -> loss=0.10&0.10 with test correct=8191/8192 (demonic laugh)...
10. Turning off early termination ("correct == total -> brake" rule) solved the remainder, however I was morally
   ready to tune weight decay of my RMSprop to reduce that tiny remaining amount of variance per Andrew Ng's tutor.
11. Fit into 1s limit case. From RMSprop source code it looks like that in the case, if weight decay is
    used, gradients are changed by ANN weights' values multiplied by the weight decay. As far as I understand, this way
    RMSprop could be made to keep weights' smaller. Another grid search with 5400 cases varying weight decay,
    momentum, learning rate, hidden layers and neurons quantity + limiting time to 1.0s gave 8 hparam cases with 100% accuracy.
    Cases with lower weight decay (0.001), higher momentum (0.8) and lower learning rate (0.01) were better.
    Finally, no weight decay worked better and the task appeared to be just about searching for the good-working
    ANN structure: not big to be fast, but not too small to be stable for each of random seeds tested. Taking a look
    on loss curve sets via tensorboard - 10 curves describing all random seed cases at a time - and choosing hparams with
    less distributed curves solved the task.
"""
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
        self.layers_number = 4
        self.layers_number_grid = [3, 4]
        self.hidden_size = 18
        self.hidden_size_grid = [14, 15, 16, 17, 18]
        self.do_batch_norm = True
        # self.do_batch_norm_grid = [False, True]
        self.activation_hidden = 'relu'  # 'tanh'
        self.activation_hidden_grid = ['relu']
        self.activation_output = 'sigmoid'
        # self.activation_output_grid = self.activations.keys()
        self.learning_rate = 0.007
        self.learning_rate_grid = [0.006, 0.007, 0.008]
        self.weight_decay = 0.
        self.weight_decay_grid = [0.0]
        self.momentum = 0.9  # 0.8
        self.momentum_grid = [0.9]
        self.random = 0
        self.random_grid = [_ for _ in range(10)]
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
        return "{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.learning_rate, self.momentum, self.hidden_size, self.activation_hidden, self.activation_output,
            self.do_batch_norm, "{0:03d}".format(self.layers_number), self.weight_decay
        )

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        key = self.get_key()
        if key in self.sols and self.sols[key] == -1:
            return
        step = 0
        loss = None
        # Put model in train mode
        model.train()
        # optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                  alpha=0.99, weight_decay=self.weight_decay, eps=1e-08, centered=False)
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
            if (
                    # correct == total or
                    time_left < 0.1 or
                    (self.grid_search.enabled and step > 100)
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
        self.time_limit = 1.0
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
        # data = data.mul(2.).add(-1.)
        # target = target.mul(2.).add(-1.)
        # print('DATA CHANGED!!!')
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
sm.SolutionManager(Config()).run(case_number=-1)
