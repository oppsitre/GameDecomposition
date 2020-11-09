import numpy as np
# from policies.utils import best_response
import random
# import matplotlib
# matplotlib.use('Agg')

from matplotlib import pyplot as plt

# def regret_calculation(basic_array):
#     z = np.zeros((3, ))
#     for x in range(3):
#         z[x] = basic_array[lose(x)] - basic_array[win(x)]
#     return z

# def best_response(basic_array) -> int:
#     return np.argmax(regret_calculation(basic_array))

# class FictitiousPlay(object):
#     def __init__(self, cumulative_act, payoff):
#         self.cumulative_act = [0, 0, 0]
#         self.payoff = payoff

#     def perceive(self, t, action, outcome, reward, terminal):
#         self.cumulative_act[outcome[1]] += 1
    
#     def best_response(self):
#         p = np.array(self.cumulative_act / np.sum(self.cumulative_act))
#         p = p.reshape(3, 1)
#         exp = self.payoff @ p
#         a = np.argmax(exp)
#         return a



#     def get_action(self, t):
#         if t > 0:
#             return best_response(self.cumulative_act)
#         return random.randint(0, 2)

# def input_game():
#     A = np.array([
#         [[0, 2, 1],
#          [1, 0, 2],
#          [2, 1, 0]],
#         [[0, 1, 2],
#          [2, 0, 1],
#          [1, 2, 0]]
#     ])

#     return A, 2, [3, 3]


# def input_game():
#     A = np.array([
#         [[0, 0.5, -0.5],
#          [-0.5, 0, 0.5],
#          [0.5, -0.5, 0]],
#         [[0, -0.5, 0.5],
#          [0.5, 0, -0.5],
#          [-0.5, 0.5, 0]]
#     ])

#     return A, 2, [3, 3]

# def input_game():
#     A = np.array([
#         [[0, 2, 1],
#          [1, 0, 2],
#          [2, 1, 0]],
#         [[0, 1, 2],
#          [2, 0, 1],
#          [1, 2, 0]]
#     ])

#     return A, 2, [3, 3]

def input_game():
    A = np.array([
        [[0, 1, -1],
         [-1, 0, 1],
         [1, -1, 0]],
        [[0, -1, 1],
         [1, 0, -1],
         [-1, 1, 0]]
    ])

    return A, 2, [3, 3]

# def input_game():
#     A = np.array([[[-1.,  0.5,  0.5],
#                    [0.5, - 1.,  0.5],
#                    [0.5,  0.5, - 1.]],
#                   [[-1.,  0.5,  0.5],
#                    [0.5,  -1.,  0.5],
#                    [0.5,  0.5,  -1.]]])
#     return A, 2, [3, 3]

# def input_game():
#     A = np.array([
#         [[-3.33066907e-16,  5.00000000e-01, -5.00000000e-01],
#          [-5.00000000e-01,  6.66133815e-16,  5.00000000e-01],
#          [5.00000000e-01,  -5.00000000e-01, -9.99200722e-16]],
#         [[-7.77156117e-16, -5.00000000e-01,  5.00000000e-01],
#          [5.00000000e-01,  2.22044605e-16,  -5.00000000e-01],
#          [-5.00000000e-01,  5.00000000e-01, -6.66133815e-16]]])
#     return A, 2, [3, 3]

# def input_game():
#     A = np.array([
#         [[2.22044605e-16,  4.44089210e-16, - 2.77555756e-16],
#         [-7.09348335e-17,  4.44089210e-16, - 2.22044605e-16],
#         [0.00000000e+00,  3.79144527e-16,  2.22044605e-16]],

#         [[4.44089210e-16, - 1.11022302e-16, - 4.44089210e-16],
#         [-4.44089210e-16,  0.00000000e+00, - 4.44089210e-16],
#         [-5.55111512e-17,  3.33066907e-16,  6.66133815e-16]]
#     ])
#     return A, 2, [3, 3]

# def input_game():
#     A = np.array([
#         [[0, 2, 1],
#          [1, 0, 2],
#          [2, 1, 0]],
#         [[0, 1, 2],
#          [2, 0, 1],
#          [1, 2, 0]]
#     ], dtype=float)

#     A[0, :, :] -= np.ones((3, 3), dtype=float)
#     A[1, :, :] -= np.ones((3, 3), dtype=float)
#     print('A', A)
#     return A, 2, [3, 3]
#     	A	B	C
# a	0, 0	2, 1	1, 2
# b	1, 2	0, 0	2, 1
# c	2, 1	1, 2	0, 0

# def input_game():
#     A = np.array([
#         [[3, 0],
#          [5, 1]],
#         [[3, 5],
#          [0, 1]],
#     ])

#     return A, 2, [2, 2]


# def input_game(theta=0.1):
#     A = np.array([
#         [[0, 1, theta / 2.0],
#          [1, 0, 0],
#          [1, 0, theta]],
#         [[1, 0, 0],
#          [0, 1, 1],
#          [0, theta / 2.0, theta]]

#     ])
#     return A, 2, [3, 3]

# def input_game():
#     A = np.array([
#         [[3, -2],
#          [-2, 1]],
#         [[-3, 2],
#          [2, -1]]
#     ])

#     return A, 2, [2, 2]

class RegretMatching:
    def __init__(self, id, n_action, payoff, count=None):
        return


class FP:
    def __init__(self, id, n_action, payoff, count=None):
        self.id = id
        self.count = count

        # if count is not None:
        #     self.count = count
        # else:
        #     self.count = np.random.rand(n_action)
        
        self.count = np.array(self.count)

        self.payoff = payoff
        self.n_action = n_action
    
    def choose_action(self, ):
        p = np.array(self.count) / np.sum(self.count)
        b = np.argmax(p)
        if self.id == 0:
            # p = p.reshape((self.n_action, 1))
            exp = self.payoff @ p.reshape(self.n_action[1], 1)
        else:
            # p = p.reshape((1, self.n_action))
            # exp = p @ self.payoff
            exp = p.reshape(1, self.n_action[0]) @ self.payoff
        a = np.argmax(exp)
        # print('Agent', self.id, 'count', self.count, 'b', b, 'p', p, 'exp', exp, 'best response', a)
        # print(self.payoff, self.payoff.shape)
        return a
    
    def update(self, my_act=None, opp_act=None, utility=None):
        # print('opp_act', opp_act)
        self.count[opp_act] += 1

def main():
    payoff, n_agent, n_action = input_game()
    # print('payoff', payoff)
    # exit()

    agent1 = FP(id=0, n_action=n_action[0], payoff=payoff[0], count=[0, 0, 1])
    agent2 = FP(id=1, n_action=n_action[1], payoff=payoff[1], count=[1, 0, 0])

    T = 1000
    poses = []
    probs1 = []
    probs2 = []
    probs3 = []
    pairs = {}
    for i in range(n_action[0]):
        for j in range(n_action[1]):
            pairs[(i, j)] = 0

    for t in range(T):
        a1 = agent1.choose_action()
        a2 = agent2.choose_action()
        poses.append((a1, a2))
        pairs[(a1, a2)] += 1

        p = np.array(agent1.count / sum(agent1.count))
        probs1.append(p[0])
        probs2.append(p[1]) 
        probs3.append(p[2])
        
        # Update 
        agent1.update(opp_act=a2)
        agent2.update(opp_act=a1)

    plt.plot(probs1)
    plt.plot(probs2)
    plt.plot(probs3)

    print('agent 0', agent1.count)
    print('agent 1', agent2.count)
    print('pairs', pairs)

    vals = []
    for i in range(n_action[0]):
        for j in range(n_action[1]):
            vals.append(pairs[(i, j)])
    print('vals', vals, np.array(vals) / np.sum(vals))
    plt.show()
        

if __name__ == '__main__':
    main()
    
