import numpy as np
import copy


def utility_normalize(payoff):
    for i in range(payoff.shape[0]):
        # print(payoff.shape)
        minV = np.min(payoff[i, :])
        maxV = np.max(payoff[i, :])
        payoff[i, :] = payoff[i, :] - minV
        # div = maxV - minV
        # div = np.max(np.abs(payoff[i, :]))
        # print('maxV', maxV, minV, maxV - minV)
        # print(payoff[i, :])
        if maxV - minV > 0:
            payoff[i, :] = payoff[i, :] / (maxV - minV)

    return payoff

def ToGTOutput(payoff, n_agent, n_action, fname):
    with open('../data/' + fname, 'w') as f:
        print(n_agent, file=f)
        print(' '.join([str(x) for x in n_action]) + '\n', file=f)
    
# def EPRS(x=1, y=2, z=3):
#     '''
#     Two-player Game
#     Extented Rock-Paper-Scissor
#     '''
#     n_agent = 2   # int
#     n_action = [3, 3]  # list
#     payoff = {}  # dict[(a_1, ..., a_N)] = value_i ((a_1, ..., a_N))
#     A = {}
#     A[(0, 0)] = 0
#     A[(0, 1)] = -3 * x
#     A[(0, 2)] =  3 * y

#     A[(1, 0)] = 3 * x
#     A[(1, 1)] = 0
#     A[(1, 2)] = -3 * z

#     A[(2, 0)] = -3 * y
#     A[(2, 1)] = 3 * z
#     A[(2, 2)] = 0

#     B = {}
#     for key in A.keys():
#         B[key] = 1 - A[key]

#     payoff = np.array([A, B])

#     return payoff, n_agent, n_action

def payoff2ld(matrix):
    print(matrix, matrix.shape)
    n_agent = matrix.shape[0]
    n_action = []
    for i in range(1, len(matrix.shape)):
        n_action.append(matrix.shape[i])

    print('n_agent', n_agent, 'n_action', n_action)
    # exit()
    payoff = []
    for i in range(n_agent):
        payoff.append({})

    def construct_payoff(aid, acts):
        if aid == n_agent:
            for i in range(n_agent):
                tmp = [i]
                tmp.extend(acts)
                payoff[i][tuple(acts)] = matrix[tuple(tmp)]
            return

        for a in range(n_action[aid]):
            acts.append(a)
            construct_payoff(aid+1, acts)
            acts.pop()

    construct_payoff(0, [])
    return payoff


def ld2payoff(ld, n_agent, n_action):
    '''
    List Dict to Payoff
    '''
    # print('ld 0', ld[0])
    # print('ld 1', ld[1])
    n_player = n_agent
    payoff = []
    for i in range(n_player):
        tmp = np.zeros(n_action)
        tmp = np.expand_dims(tmp, axis=0)
        payoff.append(tmp)

    def construct_payoff(pid, acts):
        if pid >= n_player:
            for i in range(n_player):
                idx = [0]
                idx.extend(acts)
                payoff[i][tuple(idx)] = copy.deepcopy(ld[i][tuple(acts)])
            return 

        # print('pid', pid)
        # print('n_action', n_action)
        for a in range(n_action[pid]):
            acts.append(a)
            construct_payoff(pid+1, acts)
            acts.pop()
        
    construct_payoff(0, [])

    # print(payoff[0].shape)
    payoff = np.vstack(payoff)
    # print('p0', payoff[0, :])
    # print('p1', payoff[1, :])
    # exit()
    # print('After stack', payoff.shape)
    return payoff

def EPRS(x=1, y=2, z=3):
    '''
    Two-player Game
    Extented Rock-Paper-Scissor
    '''
    n_agent = 2   # int
    n_action = [3, 3]  # list
    # payoff = {}  # dict[(a_1, ..., a_N)] = value_i ((a_1, ..., a_N))
    A = {}
    A[(0, 0)] = 0
    A[(0, 1)] = -3 * x
    A[(0, 2)] = 3 * y

    A[(1, 0)] = 3 * x
    A[(1, 1)] = 0
    A[(1, 2)] = -3 * z

    A[(2, 0)] = -3 * y
    A[(2, 1)] = 3 * z
    A[(2, 2)] = 0

    B = {}
    for key in A.keys():
        B[key] = 0 - A[key]

    payoff = [A, B]
    return payoff, n_agent, n_action


def read_pot_add_har(k):
    potential = np.load('potential.npy')
    harmonic = np.load('harmonic.npy')
    tot = potential * k + harmonic * (1.0 - k)
    n_agent = tot.shape[0]
    n_action = []
    for i in range(n_agent):
        n_action.append(tot.shape[i+1])
    return tot, n_agent, n_action

def coingame():
    '''
    Two-player Game
    Extented Rock-Paper-Scissor
    '''
    n_agent = 2   # int
    n_action = [2, 2]  # list
    # payoff = {}  # dict[(a_1, ..., a_N)] = value_i ((a_1, ..., a_N))
    A = {}
    A[(0, 0)] = 3
    A[(0, 1)] = -2
    
    A[(1, 0)] = -2
    A[(1, 1)] = 1

    B = {}
    for key in A.keys():
        B[key] = 0 - A[key]

    payoff = [A, B]
    return payoff, n_agent, n_action

def random_game(n_agent=2, n_action=[5, 5]):
    A = {}
    B = {}
    for i in range(n_action[0]):
        for j in range(n_action[1]):
            A[(i, j)] = np.random.rand()
            B[(i, j)] = np.random.rand()
    
    payoff = [A, B]
    return payoff, n_agent, n_action

def pot_add_har(k=0.5, n_agent=2, n_action=[3, 3]):
    shape = [n_agent]
    shape.extend(n_action)
    pot = np.ones(shape) * -1
    har = np.zeros(shape)
    for a in range(n_action[0]):
        for i in range(n_agent):
            pot[i, a, a] = 1
    
    for a in range(n_action[0]):
        for b in range(n_action[1]):
            
            if (b - a + 3) % 3 == 1:
                har[0, a, b] = 1
            elif (b - a + 3) % 3 == 2:
                har[0, a, b] = -1

            har[1, a, b] = 0 - har[0, a, b]
    # print('pot', pot)
    # print('har', har)
    tot = pot * k + har * (1.0 - k)
    # print('tot', tot)
    # exit()
    return tot, n_agent, n_action

def input_game_1():
    A = np.array([
        [[0.90575177, 0.49724717, 0.77514925, 0.03547901, 0.27725938],
         [0.47837718, 0.76700907, 0.01202431, 0.62039155, 0.34803773], 
         [0.79854966, 0.81840104, 0.32984793, 0.40238176, 0.67465554], 
         [0.86414042, 0.97926696, 0.81570663, 0.81464063, 1.        ], 
         [0.        , 0.52177366, 0.14808302, 0.41158384, 0.23497164]],
        [[0.37465259, 0.96317132, 0.00381973, 0.15201818, 0.61580478],
         [0.4251907 , 0.16886291, 0.62173551, 0.99892993, 0.28744749],
         [0.08053608, 0.        , 0.06510401, 0.68445911, 1.        ],
         [0.81827948, 0.65141333, 0.82655369, 0.82824173, 0.51807078],
         [0.14418299, 0.1176996 , 0.10265407, 0.35337199, 0.61925687]]
        ])

    return A, 2, [5, 5]

def input_game_4():
    A = np.array([
                 [[0.1, 0.],
                  [0.9, 0.9]],
                 [[1., 0.],
                  [1., 0.]]])
    return A, 2, [2, 2]

def input_game_6():
    A = np.array([[[0., 0.],
          [0., 0.]],
         [[0., 1.],
          [0., 1.]]])
    return A, 2, [2, 2]

def input_game_5():
    A = np.array([
        [[1.0, 1.0],
         [0.0, 0.0]],
        [[1.0, 0.0],
         [0.0, 1.0]],
    ])
    return A, 2, [2, 2]

def input_game_2():
    A = np.array(
        [
         [[0.51326467, 0.12430268, 0.15671684, 0.66665899, 0.70876102],
          [0.53829291, 0.5814566 , 1.        , 0.85120512, 0.09868475],
          [0.66487011, 0.79596564, 0.2333149 , 0.04062394, 0.52913748],
          [0.56672748, 0.65278739, 0.57209351, 0.05644815, 0.52079869],
          [0.18658807, 0.26796149, 0.        , 0.38082533, 0.59760021]],

        [[0.27514894, 0.22166985, 1.        , 0.6916646 , 0.05673938],
         [0.95571673, 0.95621442, 0.15644016, 0.58388866, 0.08006389],
         [0.10005715, 0.36011457, 0.66052953, 0.32394903, 0.42905549],
         [0.29330269, 0.14487557, 0.61965798, 0.05336979, 0.21402379],
         [0.        , 0.14758351, 0.96088714, 0.17142604, 0.07087487]]
        ])
    
    return A, 2, [5, 5]

def input_game_3():
    A = np.array([
        [[0.89690388, 0.18595268, 0.77744876, 0.66360053, 0.28204753],
         [1.        , 0.96008117, 0.34750945, 0.        , 0.71106944],
         [0.0033633 , 0.42323195, 0.23411107, 0.66790094, 0.29725051],
         [0.63518683, 0.38389942, 0.80067195, 0.69941346, 0.20546299],
         [0.68074566, 0.19563615, 0.61955989, 0.21301044, 0.11151898]],

        [[0.56671842, 0.88996875, 0.75930054, 0.68955436, 0.20051432],
         [0.27679632, 0.50327803, 1.        , 0.01545702, 0.        ],
         [0.79777582, 0.83720117, 0.99363763, 0.58589854, 0.12727307],
         [0.0143434 , 0.5661085 , 0.36066814, 0.29645518, 0.85953561],
         [0.46173079, 0.11013977, 0.87992011, 0.32408172, 0.41991752]]
        ])

    return A, 2, [5, 5]

if __name__ == '__main__':
    payoff, n_agent, n_action = input_game_3()
    print(payoff.shape)
    # payoff, n_player, n_action = EPRS(x=1, y=1, z=1)
    # payoff = ld2payoff(payoff, n_player, n_action)
    # print(payoff)
    # print(n_player)
    # print(n_action)
