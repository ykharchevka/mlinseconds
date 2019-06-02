'''
# distributions visualization script:
import numpy as np
s = np.random.uniform(0., 1., 1000)    # uniform distribution             # numpy.random.uniform(low=0.0, high=1.0, size=None)
s = np.random.rand(1000)               # uniform in [0, 1)                # numpy.random.rand(d0, d1, ..., dn)
s = np.random.lognormal(0., 1., 1000)  # log-normal distribution          # numpy.random.lognormal(mean=0.0, sigma=1.0, size=None)
s = np.random.normal(size=1000)        # Gaussian                         # numpy.random.normal(loc=0.0, scale=1.0, size=None)
s = np.random.logistic                 # logistic distribution            # numpy.random.logistic(loc=0.0, scale=1.0, size=None)
s = np.random.logseries                # logarithmic series distribution  # numpy.random.logseries(p, size=None)
import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 15, density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()

'''
import subprocess
import numpy as np

class FixMeFixer():
    def __init__(self):
        self.counter = 0
        self.best_loss = 999.
        self.best_gamma = np.array([])
        self.best_betta = np.array([])
        self.best_epsln = np.array([])

    def print_summary(self):
        print('--------------------------------------')
        print('Search interrupted')
        print('--------------------------------------')
        print('best found loss', self.best_loss)
        print('gamma', str(self.best_gamma.tolist()))
        print('betta', str(self.best_betta.tolist()))
        print('epsln', str(self.best_epsln.tolist()))

    @staticmethod
    def do_check(gamma, betta, epsln):
        gamma = [str(i) for i in gamma]
        betta = [str(i) for i in betta]
        epsln = [str(i) for i in epsln]
        proc = subprocess.Popen(
            ['python', '-m', 'mlis.problems.fixMe', ','.join(gamma), ','.join(betta), ','.join(epsln)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outs, errs = proc.communicate()
        if errs:
            return 999.
        return float(outs.decode('utf-8').split('Loss=')[1].split('\r')[0])

    def run(self):
        while True:
            gamma = 10 ** np.random.uniform(np.log10(1e-5), np.log10(1e2),
                                            8)  # sampling from 1e-5...1e2 logarithmically              # 1 = no impact -> 0...100     logarithmically
            betta = np.random.normal(0., 30.,
                                     8)  # uniforma distribution with center at 0. and std = 30. # 0 = no impact -> -100, 100   logarithmically
            epsln = 10 ** np.random.uniform(np.log10(1e-8), np.log10(1e-4),
                                            8)  # sampling from 1e-8...1e-4 logarithmically             # 0 = no impact -> 1e-8...1e-4 logarithmically

            loss = self.do_check(gamma, betta, epsln)
            self.counter += 1
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_gamma = gamma
                self.best_betta = betta
                self.best_epsln = epsln
            print('Steps done: {:>10}. Best loss so far: {:>10} '.format(self.counter, self.best_loss), end='\r')


if __name__ == '__main__':
    fmf = FixMeFixer()
    try:
        fmf.run()
    except KeyboardInterrupt:
        fmf.print_summary()

