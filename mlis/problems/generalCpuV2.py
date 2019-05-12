'''
TODOs:
1. Double check lr, momentum, dampening and ANN size with correct (statefull) version of SGD
2. And same for RMSprop
3. Do the same for Adam
4. Go through https://www.youtube.com/watch?v=1waHlpKiNyY&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc
'''
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
                nn.Tanh()  # TODO: memorize that ANNs of linear activations model only linear functions
            ])
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self._s.hidden_size),
            nn.Tanh(),
            *hidden_layers,
            # TODO !!!!!: maybe, add here couple reducing layers
            nn.Linear(self._s.hidden_size, output_size),
            nn.Sigmoid(),
        )
        for param in self.model.parameters():
            nn.init.uniform_(param, -self._s.init, +self._s.init)  # TODO: memorize that extending range brings more non-linearity, otherwise it's about using linear segment of non-linear function

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
        self.hidden_size = 280
        self.hidden_size_grid = [280]
        self.init = 1.  # 0.65
        self.init_grid = [1.]
        self.lr = 0.0003626  # 0.0007  # 0.00099 # 0.00025=fail_at_case_3 0.00028=fail_at_case_12 0.00021=fail_at_case_10
        self.lr_grid = [0.0005]  # 0.00019, 0.00029
        self.alpha = 0.99
        self.alpha_grid = [0.99]
        self.momentum = 0.7  # 0.481  # 0.33, 0.71
        self.momentum_grid = [0.7]
        # self.dampening = 0.9
        # self.dampening_grid = [0.9]
        self.grid_search = GridSearch(self).set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        # TODO: memorize - moving optimizer init out of loop fixed momentum + damping
        # optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, dampening=self.dampening, weight_decay=0, nesterov=False)
        optimizer = optim.RMSprop(model.parameters(), lr=self.lr, alpha=self.alpha, eps=1e-08, weight_decay=0, momentum=self.momentum, centered=False)
        # optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

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
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 20

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
        input_size = min(7+case, 11)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(
            case,
            Limits(),
            (data, target),
            (data, target)
        ).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=4)  # TODO: memorize to optimize starting from the highest complexity
