import numpy as np
from my_decomposition import GameDecompostion

from gen_game import payoff2ld
from EWA import EWA, BestResponse
from find_Mccs import MCC

# import matplotlib
# matplotlib.use('Agg')

from FictitiousPlay import FP
from matplotlib import pyplot as plt
from scipy import optimize as op


# def plot_test():
#     x = np.arange(1, 100).astype(np.float64)
#     y = np.power(x, -1)
#     z = np.power(x, -0.5)
#     h = np.power(x, -0.1)

#     plt.plot(x, y, label='-1')
#     plt.plot(x, z, label='-0.5')
#     plt.plot(x, h, label='-0.1')

#     plt.legend()
#     plt.show()

# plot_test()
# exit()

def input_game(n_action=3, mode='random'):
    A = np.random.randn(2, n_action, n_action)
    return A, 2, [n_action, n_action]

def denoise(P):
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            for k in range(P.shape[2]):
                if abs(P[i, j, k]) < 1e-10:
                    P[i, j, k] = 0
    return P

def calc_steps():
    # def input_game():
    #     U = np.array([
    #         [[1, 0, 0],
    #          [0, 1, 0],
    #          [0, 0, 1]],
    #         [[0, 0, 1],
    #          [1, 0, 0],
    #          [0, 1, 0]]
    #     ], dtype=float)

    #     return U, 2, [3, 3]
    
    # def input_game():
    #     A = np.array([
    #         [[0, 2, 1],
    #          [1, 0, 2],
    #          [2, 1, 0]],
    #         [[0, 1, 2],
    #          [2, 0, 1],
    #          [1, 2, 0]]
    #     ], dtype=float)

    #     return A, 2, [3, 3]
        
    U, n_agent, n_action = input_game()

    ldpayoff = payoff2ld(U)
    gd = GameDecompostion(n_agent=n_agent, n_action=n_action, payoff=ldpayoff)
    potential, harmonic, nonstra = gd.main()


    U = harmonic
    U = denoise(U)
    print('harmonic\n', U)

    a = [U[0, 0, 0], U[0, 1, 1], U[0, 2, 2]]
    b = [U[0, 1, 0], U[0, 2, 1], U[0, 0, 2]]
    c = [U[0, 2, 0], U[0, 2, 1], U[0, 1, 2]]

    A = [U[1, 0, 2], U[1, 1, 0], U[1, 2, 1]]
    B = [U[1, 0, 0], U[1, 1, 1], U[1, 2, 2]]
    C = [U[1, 0, 1], U[1, 1, 2], U[1, 2, 0]]

    up = 1
    dw = 1
    for i in range(3):
        up *= (a[i] - c[i]) * (A[i] - C[i])
        dw *= (a[i] - b[i]) * (A[i] - B[i])
    print('Up', up, 'Down', dw)


# calc_steps()
# exit()

def calc_eps(A, p, q):
    n_action = A[0, :, :].shape
    
    # p = softmax(theta)
    # q = softmax(phi)
    # print('A', A.shape, 'p', p.shape, 'q', q.shape)
    p = p.reshape(1, -1)
    q = q.reshape(1, -1)
    
    v1 = p @ A[0, :, :] @ q.T
    v2 = p @ A[1, :, :] @ q.T

    # print('v1', v1, 'v2', v2)

    eps = 0.0
    for i in range(n_action[0]):
        one = np.zeros((1, n_action[0]))
        # print('one', one.shape, 'n_action', n_action)
        one[0, i] = 1.0
        tmp = one @ A[0, :, :] @ q.T
        eps = max(tmp - v1, eps)

    for i in range(n_action[1]):
        one = np.zeros((n_action[1], 1))
        one[i, 0] = 1.0
        tmp = p @ A[1, :, :] @ one
        eps = max(tmp - v2, eps)

    return eps[0]

# def input_game(n_action=5, mode='random'):
#     A = np.array([
#         [[3, -2],
#          [-2, 1]],
#         [[-3, 2],
#          [2, -1]]
#     ])

#     return A, 2, [2, 2]

# payoff, n_agent, n_action = input_game(n_action=5)
# ldpayoff = payoff2ld(payoff)
# gd = GameDecompostion(n_agent=n_agent, n_action=n_action, payoff=ldpayoff)
# potential, harmonic, nonstra = gd.main()
# print('potential\n', potential)
# print('harmonic\n', harmonic)
# print('nonstra', nonstra)


def run_fP_test(acc = 10, T = 1000):
    # def input_game():
    #     A = np.array([
    #         [[0, 2, 1],
    #          [1, 0, 2],
    #          [2, 1, 0]],
    #         [[0, 1, 2],
    #          [2, 0, 1],
    #          [1, 2, 0]]
    #     ], dtype=float)

    #     return A, 2, [3, 3]

    def input_game(n_action=[3, 4]):
        A = np.random.rand(2, n_action[0], n_action[1])
        return A, 2, n_action
    
    # def input_game(n_action=None):
        # A = np.array([
        #     [[0, 2, 1],
        #      [1, 0, 2],
        #      [2, 1, 0]],
        #     [[0, 1, 2],
        #      [2, 0, 1],
        #      [1, 2, 0]]
        # ], dtype=float)
        return A, 2, [3, 3]

        # A = np.array([
        #     [[1, 0, 0],
        #      [0, 1, 0],
        #      [0, 0, 1]],
        #     [[0, 0, 1],
        #      [1, 0, 0],
        #      [0, 1, 0]]
        # ], dtype=float)
        # return A, 2, [3, 3]

    payoff, n_agent, n_action = input_game(n_action=[3, 3])
    ldpayoff = payoff2ld(payoff)
    gd = GameDecompostion(n_agent=n_agent, n_action=n_action, payoff=ldpayoff)
    potential, harmonic, nonstra = gd.main()
    def denoise(P):
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                for k in range(P.shape[2]):
                    if abs(P[i, j, k]) < 1e-10:
                        P[i, j, k] = 0
        return P

    potential = denoise(potential)
    print('P', potential)    
    # print('Denoise', potential)

    harmonic = denoise(harmonic)
    print('H', harmonic)

    
    p = np.ones(n_action[0]) / n_action[0]
    q = np.ones(n_action[1]) / n_action[1]

    # print('EPS', calc_eps(harmonic, p, q))
    # print('Denoise', harmonic)

    # exit()

    nonstra = denoise(nonstra)
    
    params = []

    # def GameValue(payoff):
        

    for k in range(acc + 1):
        b = k * 1.0 / acc
        # print('B', b)
        # b = 0.01
        # b = 0.2
        utility = b * potential + (1 - b) * harmonic
        print('utility\n', utility)

        # mcc = MCC(payoff=utility, n_agent=n_agent, n_action=n_action)
        # mcc.matrix2graph()
        # mcc.main()
        # exit()

        # p = [0.2005988,  0.23053892, 0.56886228] 
        # q = [0.39520958, 0.16766467, 0.43712575]
        # p = [0.31636727, 0.36327345, 0.32035928] 
        # q = [0.35828343, 0.32335329, 0.31836327]
        # eps = calc_eps(utility, p, q)
        # print('eps', eps)
        # exit()

        # utility = potential
    
        # print('potential', potential)
        # print('harmonic', harmonic)
        # print('non-strat', nonstra)
        # utility = harmonic
        # utility = potential

        # agent1 = FP(id=0, n_action=n_action, payoff=utility[0], count=np.random.rand(n_action[0]))
        # agent2 = FP(id=1, n_action=n_action,
        #             payoff=utility[1], count=np.random.rand(n_action[1]))
        
        agent1 = FP(id=0, n_action=n_action,
                    payoff=utility[0], count=[0, 1, 0])
        agent2 = FP(id=1, n_action=n_action, payoff=utility[1], count=[1, 0, 0])

        
        poses = []
        probs1 = []
        probs2 = []
        probs3 = []
        
        eps = []
        eps_max = []
        eps_min = []

        act_max = []
        act_min = []

        vals = []
        minmax1 = []
        minmax2 = []
        minmax3 = []
        
        pairs = {}
        for i in range(n_action[0]):
            for j in range(n_action[1]):
                pairs[(i, j)] = 0

        pair_lst = []
        
        tag = 0
        act_tag = 0

        for t in range(T):
            a1 = agent1.choose_action()
            a2 = agent2.choose_action()
            poses.append((a1, a2))

            if (a1, a2) not in pair_lst:
                if len(pair_lst) > 0:
                    print((pair_lst[-1]), np.array(agent1.count), 'Min', np.min(agent1.count), np.array(agent2.count), 'Min', np.min(agent2.count))
                pair_lst.append((a1, a2))

            elif (a1, a2) == pair_lst[0] and len(pair_lst) > 1:
                for idx, par in enumerate(pair_lst):
                    ratio = 1
                    if idx >= 1:
                        ratio = pairs[par] / pairs[pair_lst[idx-1]]
                    print(par, pairs[par], ratio)
                
                for key in pairs.keys():
                    pairs[key] = 0

                pair_lst = []
                pair_nums = {}
                print('\n')
                

            pairs[(a1, a2)] += 1

            q = np.array(agent1.count / sum(agent1.count))
            p = np.array(agent2.count / sum(agent2.count))
            
            # print('p', p, 'q', q)
            
            # EPS Curve
            tmp = calc_eps(utility, p, q)
            if len(eps) == 0:
                pass
            else:
                if tmp > eps[-1] and tag == 0:
                    eps_min.append([t, eps[-1]])
                    tag = 1
                elif tmp < eps[-1] and tag == 1:
                    eps_max.append([t, eps[-1]])
                    tag = 0         
            eps.append(tmp)

            # Action Prob Curve
            if len(probs1) == 0:
                pass
            else:
                if p[0] > probs1[-1] and act_tag == 0:
                    act_min.append([t, probs1[-1]])
                    act_tag = 1
                elif p[0] < probs1[-1] and act_tag == 1:
                    act_max.append([t, probs1[-1]])
                    act_tag = 0

            probs1.append(p[0])
            probs2.append(p[1])
            probs3.append(p[2])

            # Update
            agent1.update(opp_act=a2)
            agent2.update(opp_act=a1)

        # exit()

        plt.plot(probs1)
        plt.plot(probs2)
        plt.plot(probs3)

        # print('eps', eps)
        # print('eps_max', eps_max)
        # print('eps_min', eps_min)
        # print('eps', eps)
        # exit()

        # print(np.array(eps_max).shape)
        # exit()

        plt.plot(eps)
        # plt.plot(np.array(eps_max)[:, 0], np.array(eps_max)[:, 1])
        # plt.plot(np.array(eps_min)[:, 0], np.array(eps_min)[:, 1])
        plt.show()
        
        # exit()

        # def f_1(t, a, b, c):
        #     return a * (t ** c) + b

        def f_1(x, B, C):
            return np.power(x, -C) + B

        # def f_1(x, a, b, c):
        #     return np.power(x, -c)
            # return a * np.float_power(x - k, c) + b
            # return x * (a * (1 - np.exp(-b/x)) + c * np.exp(-b/x)) - x * c

        # print('eps_max', eps_max)
        
        print('eps_max')
        B, C = op.curve_fit(f_1, np.array(
            eps_max)[1:, 0], np.array(eps_max)[1:, 1], bounds=([0, 0], [1, 5]), method='trf')[0]
        plt.plot(np.array(eps_max)[1:, 0], f_1(np.array(eps_max)[1:, 0], B, C))
        A = -1

        # print('eps_min')
        # B, C = op.curve_fit(f_1, np.array(
        #     eps_min)[1:, 0], np.array(eps_min)[1:, 1], bounds=([0, 0], [1, 10]))[0]
        # A = -1

        # print('act_1_max', act_max)
        # # B, C = op.curve_fit(f_1, np.array(
        # #     act_max)[1:, 0], np.array(act_max)[1:, 1], bounds=([0., 0.], [1., 10.]))[0]
        # B, C = op.curve_fit(f_1, np.array(
        #     act_max)[1:, 0], np.array(act_max)[1:, 1])[0]
        # plt.plot(np.array(eps_max)[1:, 0], f_1(np.array(eps_max)[1:, 0], B, C))
        # A = -1

        # print('act_1_min', act_min)
        # # B, C = op.curve_fit(f_1, np.array(
        # #     act_max)[1:, 0], np.array(act_max)[1:, 1], bounds=([0., 0.], [1., 10.]))[0]
        # B, C = op.curve_fit(f_1, np.array(
        #     act_min)[1:, 0], np.array(act_min)[1:, 1], bounds=([0, 0], [10, 10]))[0]
        # # plt.plot(np.array(eps_max)[1:, 0], f_1(np.array(eps_max)[1:, 0], B, C))
        # A = -1

        # print('a', a, 'b', b, 'c', c)
        print('A', A, 'B', B, 'C', C)
        # print('B', B, 'C', C)
        params.append((b, A, B, C))

        # plt.show()
        # exit()

        # print('agent 0', agent1.count)
        # print('agent 1', agent2.count)
        # print('pairs', pairs)

        vals = []
        for i in range(n_action[0]):
            for j in range(n_action[1]):
                vals.append(pairs[(i, j)])
        # print('vals', vals, np.array(vals) / np.sum(vals))
        # plt.show()
        with open('params_eps_max.dat', 'w') as fw:
            for i, para in enumerate(params):
                # print('b', para[0], 'A', para[1], 'B', para[2], 'C', para[3], file=fw)
                print('b', para[0], 'A', para[1], 'B', para[2], 'C', para[3])
        
        # with open('params_eps_min.dat', 'w') as fw:
        #     for i, para in enumerate(params):
        #         print('b', para[0], 'A', para[1], 'B',
        #               para[2], 'C', para[3], file=fw)
        #         print('b', para[0], 'A', para[1], 'B', para[2], 'C', para[3])

        # with open('params_act_1_min.dat', 'w') as fw:
        #     for i, para in enumerate(params):
        #         print('b', para[0], 'A', para[1], 'B', para[2], 'C', para[3], file=fw)
        #         print('b', para[0], 'A', para[1], 'B', para[2], 'C', para[3])


run_fP_test(T=int(1e3), acc=10)
exit()

def run_dynamic(T, acc, agent1, agent2, payoff):
    pairs = {}
    for i in range(n_action[0]):
        for j in range(n_action[1]):
            pairs[(i, j)] = 0

    poses = []
    for t in range(T):
        a1 = agent1.choose_action()
        a2 = agent2.choose_action()
        poses.append((a1, a2))
        pairs[(a1, a2)] += 1

        # Update
        agent1.update(my_act=a1, opp_act=a2, utility=payoff[0, :, a2])
        agent2.update(my_act=a2, opp_act=a1, utility=payoff[1, a1, :])

        # print('agent 1 Q', agent1.Q[-1], 'N', agent1.N[-1])
        # print('agent 2 Q', agent2.Q[-1], 'N', agent2.N[-1])

    vals = []
    pays = [0, 0]
    for i in range(n_action[0]):
        for j in range(n_action[1]):
            vals.append(pairs[(i, j)])

    print('vals', vals, np.array(vals) / np.sum(vals))

    for i in range(n_action[0]):
        for j in range(n_action[1]):
            p = pairs[(i, j)] / np.sum(vals)
            pays[0] += p * payoff[0, i, j]
            pays[1] += p * payoff[1, i, j]

    print('Expected Payoff', pays)


def run_EWA(T, acc, payoff, n_agent, n_action, mode='fp'):
    if mode == 'fp':
        '''
        Fictitious Play
        '''
        alpha = 0
        beta = 1e5
        delta = 1
        kappa = 0
    elif mode == 'BRD':
        '''
        Best Reply Dynamic
        '''
        alpha = 1
        beta = 1e5
        delta = 1
        kappa = np.random.rand()
    elif mode == 'wsfp':
        '''
        weighted stochastic fictitious play
        '''
        alpha = np.rand.rand()
        beta = 0.1
        delta = 1
        kappa = 0
    elif mode == 'tprd':
        '''
        two-population replicator dynamics
        '''
        beta = 0.1
        alpha = 0
        delta = 1
        kappa = 0.1
    elif mode == 'rl':
        '''
        Reinforcement Learning
        '''
        delta = 0

    else:
        print('EWA Mode is Wrong!')
        exit()

    agent1 = EWA(id=0, n_action=n_action[0], alpha=0, delta=1, kappa=0.5, beta=0.1)
    agent2 = EWA(id=1, n_action=n_action[1], alpha=0, delta=1, kappa=0.5, beta=0.1)
    run_dynamic(T, acc, agent1, agent2, payoff)


def run_fp(payoff, n_agent, n_action):
    agent1 = FP(id=0, n_action=n_action[0], payoff=payoff[0, :, :], count=None)
    agent2 = FP(id=1, n_action=n_action[1],
                payoff=payoff[1, :, :].T, count=None)
    
    run_dynamic(T, acc, agent1, agent2, payoff)
    # pairs = {}
    # for i in range(n_action[0]):
    #     for j in range(n_action[1]):
    #         pairs[(i, j)] = 0

    # T = 1000
    # poses = []
    # for t in range(T):
    #     a1 = agent1.choose_action()
    #     a2 = agent2.choose_action()
    #     poses.append((a1, a2))
    #     pairs[(a1, a2)] += 1

    #     # Update
    #     agent1.update(a2)
    #     agent2.update(a1)

    # print('agent 0', agent1.count, np.array(agent1.count) / sum(agent1.count))
    # print('agent 1', agent2.count, np.array(agent2.count) / sum(agent2.count))
    # # print('pairs', pairs)

    # vals = []
    # pays = [0, 0]
    # for i in range(n_action[0]):
    #     for j in range(n_action[1]):
    #         vals.append(pairs[(i, j)])

    # print('vals', vals, np.array(vals) / np.sum(vals))

    # for i in range(n_action[0]):
    #     for j in range(n_action[1]):
    #         p = pairs[(i, j)] / np.sum(vals)
    #         pays[0] += p * payoff[0, i, j]
    #         pays[1] += p * payoff[1, i, j]
    

    # print('Expected Payoff', pays)


def run_dep(payoff, n_agent, n_action):
    ldpayoff = payoff2ld(payoff)
    gd = GameDecompostion(n_agent=n_agent, n_action=n_action, payoff=ldpayoff)
    potential, harmonic, nonstra = gd.main()
    print('potential\n', potential)
    print('harmonic\n', harmonic)
    print('nonstra', nonstra)

    print('Original')
    print((potential + harmonic) / 2)
    acc = 100
    for k in range(0, acc+1):
        b = k * 1.0 / acc
        print('B', b)
        comb = potential + (1 - b) * harmonic
        ldpayoff = payoff2ld(comb)
        gd = GameDecompostion(
            n_agent=n_agent, n_action=n_action, payoff=ldpayoff)
        potential, harmonic, nonstra = gd.main()
        # print('potential', potential)
        # print('harmonic\n', harmonic)
        # print('nonstra', nonstra)

        pays = 2 * (b * potential + (1 - b) * harmonic) + nonstra
        run_EWA(10, acc, pays, n_agent, n_action)
        mcc = MCC(payoff=pays, n_agent=n_agent, n_action=n_action)
        mcc.matrix2graph()
        mcc.main()

run_dep(payoff, n_agent, n_action)
exit()



print('EWA')
run_EWA(10, 10, payoff, n_agent, n_action, style='wsfp')
exit()

# print('FP')
# run_fp(payoff, n_agent, n_action)

# exit()

# plt.show()
print('Potential')
run_fp(potential, n_agent, n_action)

print('Harmonic')
run_fp(harmonic, n_agent, n_action)

print('Original')
acc = 10
for k in range(0, acc+1):
    b = k * 1.0 / acc
    print('B', b)
    run_fp(b * potential + (1 - b) * harmonic, n_agent, n_action)


# print('NoneS')
# run_fp(nonstra, n_agent, n_action)
