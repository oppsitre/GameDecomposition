import numpy as np
from Dynamics import FP, SFP, CFP
# from script import calc_eps
from my_decomposition import GameDecompostion
from gen_game import *
from matplotlib import pyplot as plt

# def calc_eps(A, p, q):
#     import copy
#     A = copy.deepcopy(A)
#     A = utility_normalize(A)
#     minV_1 = 1e5
#     maxV_1 = -1e5
#     minV_2 = 1e5
#     maxV_2 = -1e5

#     n_action = A[0, :, :].shape

#     # p = softmax(theta)
#     # q = softmax(phi)
#     # print('A', A.shape, 'p', p.shape, 'q', q.shape)
#     p = p.reshape(1, -1)
#     q = q.reshape(1, -1)

#     v1 = p @ A[0, :, :] @ q.T
#     v2 = p @ A[1, :, :] @ q.T

#     # print('v1', v1, 'v2', v2)

#     eps = 0.0
#     for i in range(n_action[0]):
#         one = np.zeros((1, n_action[0]))
#         # print('one', one.shape, 'n_action', n_action)
#         one[0, i] = 1.0
#         tmp = one @ A[0, :, :] @ q.T

#         eps_1 = max(tmp - v1, eps)
#         minV_1 = min(minV_1, one @ A[0, :, :] @ q.T)
#         maxV_1 = max(maxV_1, one @ A[0, :, :] @ q.T)

#     for i in range(n_action[1]):
#         one = np.zeros((n_action[1], 1))
#         one[i, 0] = 1.0
#         tmp = p @ A[1, :, :] @ one

#         eps_2 = max(tmp - v2, eps)
#         minV_2 = min(minV_2, p @ A[1, :, :] @ one)
#         maxV_2 = max(maxV_2, p @ A[1, :, :] @ one)

#     return max(eps_1, eps_2), minV_1, maxV_1, minV_2, maxV_2, eps_1 + eps_2

def input_game(n_action, theta=5, mode='ZS+II'):
    if mode == 'ZS+II':
        C = np.random.rand(2, n_action[0], n_action[1])
        C[1, :, :] = C[0, :, :]

        B = np.random.rand(2, n_action[0], n_action[1])
        B[1, :, :] = -B[0, :, :]

        A = B + C

    # elif mode == 'NI+NH':
    #     C = np.random.rand(2, n_action[0], n_action[1])
    #     C[1, :, :] = C[0, :, :]

    #     B = np.random.rand(2, n_action[0], n_action[1])
    #     B[1, :, :] = -B[0, :, :]
    if mode == 'HA+II':
        payoff = np.random.rand(2, n_action[0], n_action[1])
        ldpayoff = payoff2ld(payoff)
        gd = GameDecompostion(n_agent=2, n_action=n_action, payoff=ldpayoff)
        potential, harmonic, nonstra = gd.main()

        C = np.random.rand(2, n_action[0], n_action[1])
        C[1, :, :] = C[0, :, :]

        A  = harmonic + C

    elif mode == 'ZS':
        A = np.random.rand(2, n_action[0], n_action[1])
        A[1, :, :] = -A[0, :, :]
    
    elif mode == 'II':
        A = np.random.rand(2, n_action[0], n_action[1])
        A[1, :, :] = A[0, :, :]
        # A = utility_normalize(A)

    elif mode == 'HA':
        payoff = np.random.rand(2, n_action[0], n_action[1])
        gd = GameDecompostion(
            n_agent=2, n_action=n_action, payoff=payoff2ld(payoff))
        potential, harmonic, nonstra = gd.main()

        A = harmonic

    elif mode == 'PA':
        payoff = np.random.rand(2, n_action[0], n_action[1])
        gd = GameDecompostion(
            n_agent=2, n_action=n_action, payoff=payoff2ld(payoff))
        potential, harmonic, nonstra = gd.main()

        A = potential

    elif mode == 'random':
        A = np.random.rand(2, n_action[0], n_action[1])

    elif mode == 'NI':
        A = -np.ones(((2, n_action[0], n_action[1])))
        for i in range(n_action[0]):
            A[0, i, i] = 2
            A[0, i, i] = 2

        # A = np.array([
        #     [[2, -1, -1],
        #     [-1, 2, -1],
        #     [-1, -1, 2]],
        #     [[2, -1, -1],
        #     [-1, 2, -1],
        #     [-1, -1, 2]]
        # ], dtype=float)
        return A, 2, n_action
# 2, 2 - 1, -1 - 1, -1
# -1, -1 2, 2 - 1, -1
# -1, -1 - 1, -1 2, 2
    elif mode == 'shapely':
        print('Shapely Game')
        A = np.array([
            [[0, 2, 1],
             [1, 0, 2],
             [2, 1, 0]],
            [[0, 1, 2],
             [2, 0, 1],
             [1, 2, 0]]
        ], dtype=float)
        return A, 2, [3, 3]
    
    elif mode == 'RPS':
        print('RPS Game')
        A = np.array([
            [[0, 1, -1],
             [-1, 0, 1],
             [1, -1, 0]],
            [[0, -1, 1],
             [1, 0, -1],
             [-1, 1, 0]]
        ], dtype=float)
        # A[1, :, :] = -A[0, :, :]
        return A, 2, [3, 3]

        # A = np.array([
        # [[0, 1, theta],
        #  [theta, 0, 1],
        #  [1, theta, 0]],
        # [[0, theta, 1],
        #  [1, 0, theta],
        #  [theta, 1, 0]]
        # ], dtype=float)
        # return A, 2, [3, 3]
# 4,4 -1,1 1,-1
# 1,-1 2,2 -2,0
# -1,1 0,-2 2,2
        # A = np.array([
        # [[4, -1, 1],
        #  [1, 2, -2],
        #  [-1, 0, 2]],
        # [[4, 1, -1],
        #  [-1, 2, 0],
        #  [1, -2, 2]]
        # ], dtype=float)
        # A = np.array(
        #     [[[0.04439473, -0.06901478,  0.02462005],
        #       [0.09501553,  0.18136633, -0.27638186],
        #       [-0.13941026, -0.11235155, 0.25176181]],
        #      [[-0.04439473,  0.06901478, -0.02462005],
        #       [-0.09501553, -0.18136633,  0.27638186],
        #       [0.13941026,  0.11235155, -0.25176181]]]
        # )
        

    return A, 2, n_action


def run_fp_test(agent1, agent2, utility, T=int(1e1)):
    print('utility\n', utility)
    print(utility[0, 0, 2], utility[1, 0, 2])
    x = []
    y = []
    z = []
    h = []
    for epoch in range(T):
        print('Epoch %d' % epoch)
        
        print('Previous')
        q = np.array(agent1.count / sum(agent1.count))
        p = np.array(agent2.count / sum(agent2.count))
        print('p', agent2.count, p)
        print('q', agent1.count, q)
        tmp, minV_1, maxV_1, minV_2, maxV_2, eps_1, eps_2 = calc_eps(utility, p, q)
        print('maxV_1', maxV_1, 'expV_1', (utility[0, :, :] @ q.reshape((-1, 1))).flatten(), 'eps_1', eps_1, 'eps_2', eps_2)
        print('maxV_2', maxV_2, 'expV_2', (p.reshape((1, -1)) @
                                           utility[1, :, :]).flatten(), 'eps_1', eps_1, 'eps_2', eps_2)
        print('sum of maxV %f, sum of minV %f, sum of Eps %f' % (maxV_1 + maxV_2, minV_1 + minV_2, eps_1 + eps_2))

        a1 = agent1.choose_action()
        a2 = agent2.choose_action()
        print('a1 %d a2 %d' % (a1, a2))
        print('Player 1 Use %d Receive %f' % (a1, utility[0, a1, a2]))
        print('Player 2 Use %d Receive %f' % (a2, utility[1, a1, a2]))
        agent1.update(opp_act=a2)
        agent2.update(opp_act=a1)

        print('After')
        q = np.array(agent1.count / sum(agent1.count))
        p = np.array(agent2.count / sum(agent2.count))
        print('p', agent2.count, p)
        print('q', agent1.count, q)
        tmp, minV_1, maxV_1, minV_2, maxV_2, eps_1, eps_2 = calc_eps(
            utility, p, q)
        print('maxV_1', maxV_1, 'minV_1', minV_1, 'expV_1',
              (utility[0, :, :] @ q.reshape((-1, 1))).flatten(), 'eps_1', eps_1, 'eps_2', eps_2)
        print('maxV_2', maxV_2, 'minV_2', minV_2, 'expV_2', (p.reshape(
            (1, -1)) @ utility[1, :, :]).flatten(), 'eps_1', eps_1, 'eps_2', eps_2)
        print('sum of maxV %f, sum of minV %f, sum of Eps %f' % (maxV_1 + maxV_2, minV_1 + minV_2, eps_1 + eps_2))
        x.append(epoch)
        y.append(maxV_1 + maxV_2)
        z.append(minV_1 + minV_2)
        h.append(eps_1 + eps_2)
    
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    z = np.array(z).flatten()
    h = np.array(h).flatten()

    plt.plot(x, y, label='max')
    # plt.plot(x, z, label='min')
    # plt.plot(x, h, label='eps')
    plt.legend()
    plt.show()
    q = np.array(agent1.count / sum(agent1.count))
    p = np.array(agent2.count / sum(agent2.count))
    tmp, minV_1, maxV_1, minV_2, maxV_2, eps_1, eps_2 = calc_eps(utility, p, q)
    print('sum of maxV %f, sum of minV %f, sum of Eps %f' % (maxV_1 + maxV_2, minV_1 + minV_2, eps_1 + eps_2))
    return maxV_1 + maxV_2


def run_cfp_test(agent1, agent2, utility, T=int(1e1)):
    print('utility\n', utility)
    x = []
    y = []
    z = []
    h = []
    for epoch in range(T):
        print('Epoch %d' % epoch)

        print('Previous')
        p = agent1.choose_action()
        q = agent2.choose_action()
        print('p', agent1.count, agent1.T, p)
        print('q', agent2.count, agent2.T, q)
        tmp, minV_1, maxV_1, minV_2, maxV_2, eps_1, eps_2 = calc_eps(
            utility, p, q)
        print('maxV_1', maxV_1, 'expV_1',
              (utility[0, :, :] @ q.reshape((-1, 1))).flatten(), 'eps_1', eps_1, 'eps_2', eps_2)
        print('maxV_2', maxV_2, 'expV_2', (p.reshape((1, -1)) @
                                           utility[1, :, :]).flatten(), 'eps_1', eps_1, 'eps_2', eps_2)
        print('sum of maxV %f, sum of minV %f, sum of Eps %f' %
              (maxV_1 + maxV_2, minV_1 + minV_2, eps_1 + eps_2))

        agent1.update(opp_act=q)
        agent2.update(opp_act=p)
        
        a1 = agent1.grad
        a2 = agent2.grad

        print('a1 %d a2 %d' % (a1, a2))
        print('Player 1 Use %d Receive %f' % (a1, utility[0, a1, a2]))
        print('Player 2 Use %d Receive %f' % (a2, utility[1, a1, a2]))
        # agent1.update(opp_act=a2)
        # agent2.update(opp_act=a1)

        print('After')
        p = agent1.choose_action()
        q = agent2.choose_action()
        print('p', agent1.count, agent1.T, p)
        print('q', agent2.count, agent2.T, q)
        tmp, minV_1, maxV_1, minV_2, maxV_2, eps_1, eps_2 = calc_eps(
            utility, p, q)
        print('maxV_1', maxV_1, 'minV_1', minV_1, 'expV_1',
              (utility[0, :, :] @ q.reshape((-1, 1))).flatten(), 'eps_1', eps_1, 'eps_2', eps_2)
        print('maxV_2', maxV_2, 'minV_2', minV_2, 'expV_2', (p.reshape(
            (1, -1)) @ utility[1, :, :]).flatten(), 'eps_1', eps_1, 'eps_2', eps_2)
        print('sum of maxV %f, sum of minV %f, sum of Eps %f' %
              (maxV_1 + maxV_2, minV_1 + minV_2, eps_1 + eps_2))
        x.append(epoch)
        y.append(maxV_1 + maxV_2)
        z.append(minV_1 + minV_2)
        h.append(eps_1 + eps_2)

    x = np.array(x).flatten()
    y = np.array(y).flatten()
    z = np.array(z).flatten()
    h = np.array(h).flatten()

    plt.plot(x, y, label='max')
    # plt.plot(x, z, label='min')
    # plt.plot(x, h, label='eps')
    plt.legend()
    plt.show()

    q = np.array(agent1.count / sum(agent1.count))
    p = np.array(agent2.count / sum(agent2.count))
    tmp, minV_1, maxV_1, minV_2, maxV_2, eps_1, eps_2 = calc_eps(utility, p, q)
    print('sum of maxV %f, sum of minV %f, sum of Eps %f' %
          (maxV_1 + maxV_2, minV_1 + minV_2, eps_1 + eps_2))
    return maxV_1 + maxV_2

if __name__ == '__main__':
    n_action = [3, 3]
    P0 = np.zeros(n_action[1])
    P0[0] = 1
    Q0 = np.zeros(n_action[0])
    Q0[1] = 1

    A, n_agent, n_action = input_game(n_action, mode='shapely')
    # A, n_agent, n_action = input_game(n_action=[3, 3], mode='ZS')
    # ldpayoff = payoff2ld(payoff)
    gd = GameDecompostion(
        n_agent=n_agent, n_action=n_action, payoff=payoff2ld(A))

    # potential, harmonic, nonstra = gd.main()
    # A, n_agent, n_action = input_game(n_action, mode='RPS')
    P, H, N = potential, harmonic, nonstra = gd.main()
    print('Potential\n', denoise(P))
    print('Harmonic\n', denoise(H))
    print('NonStrat\n', denoise(N))

    # P, n_agent, n_action = input_game(n_action, mode='ZS')
    # print('P\n', P)


    T = int(2e1)
    # ratios = range(0, 10)
    ratios = [0.5]
    x = []
    y = []
    for ratio in ratios:
        # ratio /= 10.0
        print('Ratio ', ratio)
        A = ratio * P + (1 - ratio) * H
        agent1 = CFP(id=0, n_action=n_action, payoff=A[0], count=P0)
        agent2 = CFP(id=1, n_action=n_action, payoff=A[1], count=Q0)
        res = run_cfp_test(agent1, agent2, A, T)
        
        # agent1 = FP(id=0, n_action=n_action, payoff=A[0], count=P0)
        # agent2 = FP(id=1, n_action=n_action, payoff=A[1], count=Q0)
        # res = run_fp_test(agent1, agent2, A, T)

        x.append(ratio)
        y.append(res)
    
    # x = np.array(x).flatten()
    # y = np.array(y).flatten()
    # plt.plot(x, y)
    # plt.show()

