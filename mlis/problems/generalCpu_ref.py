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
                # TODO: memorize that ANNs of linear activations model only linear functions
                nn.Tanh()
            ])
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self._s.hidden_size),
            nn.Tanh(),
            *hidden_layers,
            nn.Linear(self._s.hidden_size, output_size),
            nn.Sigmoid(),
        )
        for param in self.model.parameters():
            # TODO: memorize that extending range brings more non-linearity, otherwise it's about using linear segment of non-linear function
            nn.init.uniform_(param, -self._s.init, +self._s.init)

    def forward(self, x):
        return self.model.forward(x)

    def calc_loss(self, output, target):
        loss_fn = nn.BCELoss()
        loss = None
        try:
            loss = loss_fn(output, target)
        except RuntimeError:
            print('Runtime error at hidden depth: {}, hidden size: {}, lr: {}, alpha: {}, momentum: {}'.
                  format(self._s.hidden_depth, self._s.hidden_size, self._s.lr, self._s.alpha, self._s.momentum))
            exit()
        return loss

    def calc_predict(self, output):
        predict = output.round()
        return predict

class Solution():
    def __init__(self):
        # TODO: memorize probable correct approach to hparams tuning: start with default params and first define ANN structure that fully learns
        self.hidden_depth = 3
        self.hidden_depth_grid = [3]
        self.hidden_size = 50
        self.hidden_size_grid = [50]
        self.init = 1.
        self.init_grid = [1.]
        self.lr = 0.001
        self.lr_grid = [0.001]
        self.alpha = 0.99
        self.alpha_grid = [0.99]
        self.momentum = 0.7
        self.momentum_grid = [0.7]
        self.grid_search = GridSearch(self).set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        # TODO: memorize - moving optimizer init out of loop fixed momentum + damping
        optimizer = optim.RMSprop(model.parameters(), lr=self.lr, alpha=self.alpha, eps=1e-08, weight_decay=0, momentum=self.momentum, centered=False)

        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1 or step >= 10:
                break
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
            self.grid_search.log_step_value('loss', loss.item(), step)
            self.grid_search.log_step_value('ratio', total / (correct + 1e-8), step)
            if correct == total:
                break
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # update model: model.parameters() -= lr * gradient
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
# TODO: memorize to optimize starting from the highest complexity
sm.SolutionManager(Config()).run(case_number=-1)
