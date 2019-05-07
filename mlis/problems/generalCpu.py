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
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self._s = solution
        # sm.SolutionManager.print_hint("Hint[1]: Explore more deep neural networks")
        hidden_layers = []
        for i in range(self._s.hidden_depth):
            hidden_layers.extend([
                nn.Linear(self._s.hidden_size, self._s.hidden_size),
                nn.LeakyReLU()
            ])
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self._s.hidden_size),
            nn.LeakyReLU(),
            *hidden_layers,
            nn.Linear(self._s.hidden_size, output_size),
            nn.Sigmoid(),
        )
        for param in self.model.parameters():
            nn.init.uniform_(param, -1.0, +1.0)

    def forward(self, x):
        return self.model.forward(x)

    def calc_loss(self, output, target):
        loss_fn = nn.BCELoss()
        loss = None
        try:
            loss = loss_fn(output, target)
        except RuntimeError:
            print(
                'Runtime error occurred with hidden_depth={}, hidden_size={}, lr={}'
                .format(self._s.hidden_depth, self._s.hidden_size, self._s.lr)
            )
        return loss

    def calc_predict(self, output):
        predict = output.round()
        return predict

class Solution():
    def __init__(self):
        self.hidden_depth = 2
        self.hidden_depth_grid = [1, 2, 3, 4, 5, 10, 15]
        self.hidden_size = 16
        self.hidden_size_grid = [4, 8, 16, 32, 64, 128, 256]
        self.lr = 0.01
        self.lr_grid = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
        self.alpha = 0.999
        self.alpha_grid = [0.999, 0.99, 0.98, 0.97, 0.96, 0.95]
        self.momentum = 0.
        self.momentum_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.grid_search = GridSearch(self).set_enabled(True)  # TODO: set to True, if grid search needed

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1 or step >= 100:
                break
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=self.lr,
                alpha=self.alpha,
                weight_decay=.0,
                momentum=self.momentum,
                eps=1e-08,
            )
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            # sm.SolutionManager.print_hint("Hint[2]: Explore other activation functions", step)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # calculate loss
            # sm.SolutionManager.print_hint("Hint[3]: Explore other loss functions", step)
            loss = model.calc_loss(output, target)
            if loss is None:
                correct = 0
                break
            if correct == total:
                self.print_stats(step, loss.item(), correct, total)
                break
            self.grid_search.log_step_value('loss', loss.item(), step)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step

    def print_stats(self, step, loss, correct, total):
        print(
            "Step = {:>3} Prediction = {:>2}/{:>2} Error = {:>20} if hidden_depth={:>3}, hidden_size={:>3}, lr={:>10}, alpha={:>5}, momentum={:>3}"
            .format(step, correct, total, loss, self.hidden_depth, self.hidden_size, self.lr, self.alpha, self.momentum)
        )

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
sm.SolutionManager(Config()).run(case_number=10)
