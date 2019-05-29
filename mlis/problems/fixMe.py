# There are solution for generalCpu in 10 steps.
#
# BUT unfortunetly, someone made mistake and disabled batch normalization.
# See "FIX ME"
#
# You can not simple fix mistake, you can change only activation function at this point

'''
https://www.youtube.com/watch?v=tNIpEZLv_eg
https://www.youtube.com/watch?v=em6dfRxYkYU
https://www.youtube.com/watch?v=nUUqwaxLnWs
https://www.youtube.com/watch?v=5qefnAek8OA
_____________________
0. https://github.com/romankoshlyak/mlinseconds/commit/564113a7bc7f05fb286d77dbdf38ad387732c4c5
 - Solution: check out BatchInitialization and Debug. BatchInit - batch normalization but from init point of view
____________________________________________________________________________________________________
0. https://gist.github.com/SergiiKashubin/79d5b1576df11642103d671f3a0dda1d/revisions

1. I tried to convert the input data into "number of times each feature was 1 for a voter" with one-hot encoding, e.g. if a voters are [001, 101, 000] than feature times would be [1, 0, 2] and network  input would be [010 100 001]. But processing the data was taking too much time for that and I couldn't learn anything useful in a time limit

2. I used a network from findMe as a predictor for a single voter and combined the results as a simple sum + bias -> sigmoid. My idea behind this was that the network output should not change if the voters in the input are permuted. I've also added minibatch on top which helped immediately and the model started working on first 2-3 tests.

3. At this point I was passing a few first tests and others were ~7-8k correct on test. I spend a lot of hours on hyperparameter tuning, trying to reduce the size of the network, change activation functions and numbers of layers, add weight initialization, look at the distibution of the learned weights and initialize to the similar distribution, add schedulers for the learning rate, I was stuck and asked a hint.

4. A hint was to use .view method for optimizing the speed. I used .view and .narrow where possible instead of copying around parts of the data, this helped but was not sufficient.

5. I noticed that while creating the model I used dict {'prelu' : nn.PReLU(), 'tanh' : nn.Tanh()} etc. to choose activation functions in grid search and thus I was using the same nn.PReLU() object for all layers, meaning it has only one parameter to learn instead of num_layers. Changed that and it helped a bit.

6. I realized that my model can be expressed in terms of a convolutional neural network, so I used Conv layers and that gave significant speedup over manual looping through the voters. After this I was reproducibly passing 8 tests.

7. I tried decreasing the learning rate when loss is small to prevent overjumping , which helped a little bit.

8. Finally I changed RMSprop to SGD and re-tuned the learning rate, momentum and weight decay. This made it - the model started to learn fast enough to get to 10 tests passed.

The biggest annoyance was the automatic CPU throttling on my laptop - sometimes a test run for 300 steps in 2s and sometimes for 100, the changes becoming more erratic with overheated CPU. This was throwing all tuning through the window sometimes - when different runs of the same test had 5x difference in steps on the same parameters.

Key Takeaways:
1. Built-in methods are probably vectorized better than naive manual solutions (kinda obvious).
2. SGD can be much faster than more sophisticated methods if tuned.
3. Knowing PyTorch library details helps in optimizations.
4. Grid search won't give you huge improvements in performance - it is worth looking for something else to change, more fundamental than just hyperparams.
Roman: your 6 - this what view hint was for, replace loop with view
 _____________________
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
'''

import math
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs


# Note: activation function should be element independent
# See: check_independence method
class MyActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


###
###
### Don't change code after this line
###
###
class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.do_norm = solution.do_norm
        layers_size = [input_size] + [solution.hidden_size] * solution.layers_number + [output_size]
        self.linears = nn.ModuleList([nn.Linear(a, b) for a, b in zip(layers_size, layers_size[1:])])
        if self.do_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(a, track_running_stats=False) for a in layers_size[1:]])

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.do_norm:
                x = self.batch_norms[i](x)
            if i != len(self.linears) - 1:
                x = MyActivation.apply(x)
            else:
                x = torch.sigmoid(x)
        return x

    def calc_loss(self, output, target):
        loss = nn.BCELoss()
        return loss(output, target)

    def calc_predict(self, output):
        predict = output.round()
        return predict


class Solution():
    def __init__(self):
        self.learning_rate = 0.05
        self.momentum = 0.8
        self.layers_number = 3
        self.hidden_size = 30
        # FIX ME:) But you can change only activation function
        self.do_norm = False
        self.layers_number = 8
        # self.learning_rate_grid = [0.04, 0.05, 0.06]
        # self.momentum_grid = [0.799, 0.8, 0.801]
        # self.layers_number_grid = [1,2,3,4,5,6,7,8]
        # self.hidden_size_grid = [20, 30, 40]
        # self.do_norm_grid = [True, False]
        self.iter = 0
        self.iter_number = 100
        self.grid_search = gs.GridSearch(self).set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # activation should be independent
    def check_independence(self):
        ind_size = 100
        for i in range(ind_size + 1):
            x = torch.FloatTensor(ind_size).uniform_(-10, 10)
            same = MyActivation.apply(x)[:i] == MyActivation.apply(x)[:i]
            print(same)
            assert same.long().sum() == i, "Independent function only"

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        self.check_independence()
        step = 0
        # Put model in train mode
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                step = math.inf
                break
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
            if total == correct:
                break
            # calculate loss
            loss = model.calc_loss(output, target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            self.print_stats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        if self.grid_search.enabled:
            self.grid_search.add_result('step', step)
            if self.iter == self.iter_number - 1:
                print(self.grid_search.choice_str, self.grid_search.get_stats('step'))
        return step

    def print_stats(self, step, loss, correct, total):
        if step % 100 == 0:
            print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, loss.item()))


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
sm.SolutionManager(Config()).run(case_number=10)
