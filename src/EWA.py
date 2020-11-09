import numpy as np

def softmax(w):
    w -= np.max(w)
    return np.exp(w) / np.sum(np.exp(w))
    
# def softmax(w):
#     return np.exp(w) / np.sum(np.exp(w))

class EWA:
    def __init__(self, id, n_action, alpha=0, kappa=1, delta=1, beta=10000):
        self.id = id
        self.t = 0
        self.n_action = n_action
        self.alpha = alpha
        self.kappa = kappa
        self.delta = delta
        self.beta = beta
        self.N = [0]
        self.cnt_step = 0
        self.Q = [[0.0] * n_action]
    
    def choose_action(self):
        Q_ = np.array(self.Q[-1]) * self.beta
        p = softmax(Q_)
        # p = Q_ - np.min(Q_) / np.sum(Q_ - np.min(Q_))
        a = np.random.choice(self.n_action, p=p)
        # print('ID', self.id, 'P', p)
        return a
    
    def update(self, my_act=None, opp_act=None, utility=None):
        N = (1 - self.alpha) * (1 - self.kappa) * self.N[-1] + 1
        Q = np.zeros(self.n_action)
        for i in range(self.n_action):
            reward = utility[i]
            Q[i] = (1 - self.alpha) * self.N[-1] * self.Q[-1][i] + \
                (self.delta + (1 - self.delta) * (my_act == i)) * reward

            Q[i] /= N
        
        self.Q.append(Q)
        self.N.append(N)

        if len(self.Q) > 2:
            self.Q = self.Q[-2:]
            self.N = self.N[-2:]            

class BestResponse:
    def __init__(self, n_action, payoff):
        self.n_action = n_action
        self.opp_action = -1
        self.payoff = payoff
    
    def choose_action(self):
        if self.opp_action == -1:
            return np.random.choice(self.n_action)
        
        utility = self.payoff[:, self.opp_action]
        return np.max(utility)
    
    def update(self, my_action=None, opp_action=None):
        self.opp_action = opp_action

        





    

    
    

