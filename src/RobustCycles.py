import numpy as np
from Dynamics import CFP, FP, SCFP
from my_decomposition import GameDecompostion
from gen_game import payoff2ld, utility_normalize, normalized_game
from matplotlib import pyplot as plt
from script import calc_eps
from scipy import optimize as op
import time

np.random.seed(int(time.time()))

class RobustCycle:
    def __init__(self, n_action, payoff, P0=None, Q0=None):
        self.n_action = n_action
        self.payoff = payoff
        self.P0 = P0
        self.Q0 = Q0

        if self.P0 is None:
            self.P0 = np.random.rand(n_action[0])
        
        if self.Q0 is None:
            self.Q0 = np.random.rand(n_action[1])
        
                
        self.P0 = np.zeros(n_action[1])
        self.P0[1] = 1

        self.Q0 = np.zeros(n_action[0])
        self.Q0[0] = 1

        # self.agent1 = FP(id=0, n_action=n_action,
        #                   payoff=payoff[0, :, :], count=self.P0)
        # self.agent2 = FP(id=1, n_action=n_action,
        #                   payoff=payoff[1, :, :], count=self.Q0)
        
        self.agent1 = FP(id=0, n_action=n_action,
                          payoff=payoff[0, :, :], count=self.P0)
        self.agent2 = FP(id=1, n_action=n_action,
                         payoff=payoff[1, :, :], count=self.Q0)

    def find_RC(self, T=int(1e3)):
        cycles = []
        for t in range(T):
            a1 = self.agent1.choose_action()
            a2 = self.agent2.choose_action()
            q = np.array(self.agent1.count / np.sum(self.agent1.count))
            p = np.array(self.agent2.count / np.sum(self.agent2.count))

            # a1 = self.agent1.grad
            # a2 = self.agent2.grad
            
            # p = self.agent1.choose_action()
            # q = self.agent2.choose_action()


            # print('p', P, 'q', Q)
            # print('a1', a1, 'a2', a2)

            # if len(cycles) == 0 or (a1, a2) != cycles[-1]:
            #     if (a1, a2) != cycles[0]:
            #         cycles.append((a1, a2))
            #     else:
            #         K = len(cycles)
            # if len(cycles) > 0:
            #     print('a1 a2', a1, a2)
            #     print(cycles[-1][0])

            if len(cycles) == 0 or (a1, a2) != cycles[-1][0]:
                cycles.append(((a1, a2), (p, q)))
            
            
            self.agent1.update(my_act=a1, opp_act=a2)
            self.agent2.update(my_act=a2, opp_act=a1)
            
            # self.agent1.update(my_act=p, opp_act=q, utility=None)
            # self.agent2.update(my_act=q, opp_act=p, utility=None)
        
        print('cycles', cycles[-10:])
        
        rc, tag = self.findCycle(cycles)
        if tag is False:
            print('RC not Found!')
            print('Cycles', rc[-10:])
            return False, rc[-10:]

        C, D = self.constructCD(rc)
        print('C', C)
        print('D', D)
        M = np.linalg.inv(C) @ D
        w, v = np.linalg.eig(M)
        print('w', w, w.shape)
        print('v', v, v.shape)

        ansW = 1
        # ansV = 1
        for b in w[1:]:
            ansW *= b
            # ansW *= w[i]
        
        print('ansW', ansW)
        return [True, rc]
        
    def calc_eps_rc(self, cycles):
        Eps = []
        MaxEps = -1
        MaxVly = -1
        MinEps = 100
        MinVly = 100
        
        for cyc in cycles:
            p, q = cyc[1]
            eps, minV_1, maxV_1, minV_2, maxV_2, vly = calc_eps(self.payoff, p, q)
            Eps.append((eps, vly))
            MaxEps = max(eps, MaxEps)
            MaxVly = max(vly, MaxVly)

            MinEps = min(eps, MinEps)
            MinVly = min(vly, MinVly)

        print('MaxEps', MaxEps, 'MaxVly', MaxVly, 'MinEps', MinEps, 'MinVly', MinVly)
        print('EPS', Eps)

        return MaxEps, MaxVly, MinEps, MinVly
    
    # def findCycle(self, cycles):
    #     s = 0
    #     e = 0
    #     while s < len(cycles):
    #         e = s + 1
    #         while e < len(cycles) and cycles[e][0] != cycles[s][0]:
    #             # print('e', e, cycles[e][0])
    #             # print('s', s, cycles[s][0])
    #             e += 1

    #         # print('s', s, 'e', e)
    #         if e >= len(cycles) or cycles[e][0] != cycles[s][0]:
    #             s += 1
    #             continue
            

    #         tag = True
    #         for k in range(s, e):
    #             if (k + e - s >= len(cycles)) or (cycles[k][0] != cycles[k + e - s][0]):
    #                 tag = False
    #                 break
            
    #         if tag is True:
    #             for k in range(s, e):
    #                 print(cycles[k])

    #             return cycles[s:e], True

    #         s += 1
            
    #     return cycles[16:], False
    
    def findCycle(self, cycles, repeat=False):
        s = len(cycles) - 1
        e = s - 1
        print('len of cycles', len(cycles))
        while s >= 0:
            while e >= 0:
                if cycles[e][0] == cycles[s][0]:
                    break
                e -= 1

            # Find a same pair
            if e >= 0: 
                break
            
            # Not Find but continue
            if e < 0 and repeat is True:
                s -= 1
                continue
            else:
                # Or stop at once.
                print('Cycles Not Found!')
                return cycles[16:], False

            # if tag is False:
            #     s -= 1
            #     continue
            # print('Robust Cycles Not Found!')
            # return cycles[16:], False
        
        # Check the cycle whether validate.
        tag = True
        for k in range(s, e, -1):
            l = k - (s - e)
            if (l < 0) or (cycles[k][0] != cycles[l][0]):
                tag = False
                break
            

        if tag is True:
            # print('Find Cycles')
            for k in range(e+1, s+1):
                print(cycles[k])
            
            # exit()
            return cycles[e+1:], True
        
        return cycles[16:], False
    
    def findBRC(self, inital_pair=None):
        R = []
        if inital_pair is None:
            cycles = [((0, 1), (None, None))]
        else:
            cycles = [(inital_pair, (None, None))]
        step = 0
        while step < 20:
            a1, a2 = cycles[-1][0]
            b1 = np.argmax(self.payoff[0, :, a2])
            b2 = np.argmax(self.payoff[1, a1, :])
            print('b1 %d b2 %d' % (b1, b2))
            if b1 != a1:
                R.append(1)
            else:
                R.append(2)

            if (a1, a2) == (b1, b2):
                break
            cycles.append(((b1, b2), (None, None)))

            step += 1

        return cycles, R

    # def findCycle(self, cycles):
    #     print('len of cycles', len(cycles))
    #     s = len(cycles) - 1
    #     while s >= 0:
    #         e = s - 1
    #         while e >= 0:
    #             if cycles[e][0] == cycles[s][0]:
    #                 break
    #             e -= 1

    #         if e < 0:
    #             s -= 1
    #             continue

    #         # print('s', s, 'e', e, 'Tag', tag)
    #         # print('Robust Cycles Not Found!')
    #         # return cycles[16:], False

    #         tag = True
    #         for k in range(s, e, -1):
    #             l = k - (s - e)
    #             if (l < 0) or (cycles[k][0] != cycles[l][0]):
    #                 tag = False
    #                 break

    #         if tag is True:
    #             print('Find Cycles')
    #             for k in range(e+1, s+1):
    #                 print(cycles[k])

    #             return cycles[e+1:s+1], True

    #         s -= 1

    #     return cycles[16:], False

    def constructCD(self, cycles):
        K = len(cycles)
        # print('K', K, cycles) 
        R = []
        for i in range(K):
            if cycles[(i+1) % K][0][0] != cycles[i % K][0][0]:
                R.append(1)
            else:
                R.append(2)
        
            # if len(R) >= 2 and R[-1] == R[-2]):
            #     print('')

        print('R', R)
        
        C = np.zeros((K, K))
        D = np.zeros((K, K))
        A = self.payoff[0, :, :]
        B = self.payoff[1, :, :]

        for k in range(0, K):
            for i in range(0, k+1):
                if R[k] == 1:
                    C[k, i] = A[cycles[(k+1)%K][0][0], cycles[i][0][1]] - \
                        A[cycles[k][0][0], cycles[i][0][1]]
                else:
                    C[k, i] = B[cycles[i][0][0], cycles[(k+1)%K][0][1]] - \
                        B[cycles[i][0][0], cycles[k][0][1]]

            for i in range(k+1, K):
                if R[k] == 1:
                    D[k, i] = -(A[cycles[(k+1)%K][0][0], cycles[i][0][1]] -
                        A[cycles[k][0][0], cycles[i][0][1]])
                else:
                    D[k, i] = -(B[cycles[i][0][0], cycles[(k+1)%K][0][1]] -
                        B[cycles[i][0][0], cycles[k][0][1]])
        
        return np.array(C), np.array(D)



def run_findRC(n_action, payoff, T=int(1e3)):
    RC = RobustCycle(n_action=n_action, payoff=payoff)
    res = RC.find_RC(T)
    if res[0] is False:
        return [False, res[1]]

    MaxEps, MaxVly, MinEps, MinVly = RC.calc_eps_rc(res[1])
    return [MaxEps, MaxVly, MinEps, MinVly]

def run_FPs(n_action, utility, count1=None, count2=None, T=int(1e3), ratio=-1):
    if count1 is None:
        count1 = np.random.rand(n_action[1])
    
    if count2 is None:
        count2 = np.random.rand(n_action[0])

    agent1 = FP(id=0, n_action=n_action,
                    payoff=utility[0], count=count1)
    agent2 = FP(id=1, n_action=n_action,
                    payoff=utility[1], count=count2)
    
    poses = []
    probs1 = []
    probs2 = []
    probs3 = []
    
    eps = []
    eps_max = []
    eps_min = []

    Vs = []
    counts = []

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

        q = np.array(agent1.count / sum(agent1.count))
        p = np.array(agent2.count / sum(agent2.count))
        # p = agent1.p
        # q = agent2.p
        # p = np.array(agent1.expected_reward / sum(agent1.expected_reward))
        # q = np.array(agent2.expected_reward / sum(agent2.expected_reward))

        # EPS Curve
        tmp, minV_1, maxV_1, minV_2, maxV_2, vly = calc_eps(utility, p, q)
        if len(eps) == 0:
            pass
        else:
            if tmp > eps[-1][1] and tag == 0:
                eps_min.append([t, eps[-1][1]])
                tag = 1
            elif tmp < eps[-1][1] and tag == 1:
                eps_max.append([t, eps[-1][1]])
                tag = 0         
        eps.append([t, tmp])
        Vs.append([t, vly])

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
        # agent1.update(opp_act=a2)
        # agent2.update(opp_act=a1)
        agent1.update(my_act=a1, opp_act=a2, utility=utility[0, :, a2])
        agent2.update(my_act=a2, opp_act=a1, utility=utility[1, a1, :])

    # exit()
    q = np.array(agent1.count / sum(agent1.count))
    p = np.array(agent2.count / sum(agent2.count))
    
    eps_final, minV_1, maxV_1, minV_2, maxV_2, vly = calc_eps(utility, p, q)

    # print('cof', , 'p', p, 'q', q)
    print('EPS_FINAL', eps_final, 'minVal_1', minV_1, 'maxVal_1', maxV_1,
            'minVal_2', minV_2, 'maxVal_2', maxV_2, 'V Function', vly)
    # exit()

    # states.append((b, p, q, minV_1, maxV_1, minV_2, maxV_2))
    
    # plt.figure()
    plt.plot(probs1, label='a1', color='b')
    plt.plot(probs2, label='a2', color='y')
    plt.plot(probs3, label='a3', color='g')

    # print('eps', eps)
    # print('eps_max', eps_max)
    # print('eps_min', eps_min)
    # print('eps', eps)
    # exit()

    # print(np.array(eps_max).shape)
    # exit()
    plt.plot(np.array(eps)[:, 0], np.array(eps)[:, 1], label='eps', color='r')
    plt.plot(np.array(Vs)[:, 0], np.array(Vs)[:, 1], label='VF', color='m')
    # plt.plot(np.array(eps_max)[:, 0], np.array(eps_max)[:, 1])
    # plt.plot(np.array(eps_min)[:, 0], np.array(eps_min)[:, 1])
    
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Probability OR Utility')
    # plt.show()
    
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

    print('eps')
    B, C = op.curve_fit(f_1, np.array(
        eps)[1:, 0], np.array(eps)[1:, 1], bounds=([0, 0], [1, 5]), method='trf')[0]
    plt.plot(np.array(eps)[1:, 0], f_1(np.array(eps)[1:, 0], B, C))
    A = -1

    # print('eps_max')
    # B, C = op.curve_fit(f_1, np.array(
    #     eps_max)[1:, 0], np.array(eps_max)[1:, 1], bounds=([0, 0], [1, 5]), method='trf')[0]
    # plt.plot(np.array(eps_max)[1:, 0], f_1(np.array(eps_max)[1:, 0], B, C))
    # A = -1

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
    plt.title('%.4fP + %.4fH, y = x^(-%.4f) + %.4f' % (ratio, 1 - ratio, C, B))
    # params.append((b, A, B, C))

    # plt.savefig('figures/zs_lambda_%.2f.png' % (b))
    # plt.show()
    # exit()

    # print('agent 0', agent1.count)
    # print('agent 1', agent2.count)
    # print('pairs', pairs)

    # for i in range(n_action[0]):
    #     for j in range(n_action[1]):
    #         vals.append(pairs[(i, j)])
    # print('vals', vals, np.array(vals) / np.sum(vals))
    # plt.show()


def run_BRC_method(n_action, payoff1, payoff2, ratios=None, T_RC=None):
    x = []
    rc = None
    for ratio in ratios:
        print('Potential\n', payoff1)
        print('Harmonic\n', payoff2)
        
        print('Ratio %.4f P %.4f H' % (ratio, 1 - ratio))
        utility = ratio * payoff1 + (1 - ratio) * payoff2
        print('Utility\n', utility)

        RC = RobustCycle(n_action, utility)
        if rc is None:
            tag, rc = RC.find_RC(T_RC)

        if tag is False:
            print('Cycles Not Found!')
            exit()

        # print('rc\n', rc)

        # RC.constructCD(rc)
        C, D = RC.constructCD(rc)
        print('C', C)
        print('D', D)
        M = np.linalg.inv(C) @ D
        w, v = np.linalg.eig(M)
        print('w', w, w.shape)
        print('v', v, v.shape)
        ansW = 1
        for b in w[1:]:
            ansW *= b
        print('ansW', ansW)
        x.append(ratio)

def run_matrix_method(n_action, payoff1, payoff2, ratios=None, T_RC=None):
    # potential *= 0.1

    # potential = utility_normalize(potential)
    # harmonic = utility_normalize(harmonic)

    # acc = 10
    # for k in range(acc + 1):
    #     ratio = k * 1.0 / acc
    #     print('Ratio %.4f P %.4f H' % (ratio, 1 - ratio))
    #     utility = ratio * potential + (1 - ratio) * harmonic
    #     # utility = utility_normalize(utility)

    #     print('Utility', utility)
    #     print('Potential', ratio * potential)
    #     print('Harmonic', (1 - ratio) * harmonic)
    #     run_findRC(n_action, utility, T=int(1e7))
        
        # p = np.array([0.57336197, 0.30310057, 0.12353746])
        # q = np.array([0.24693794, 0.34533383, 0.40772823])
        # p = np.array([0.33094281, 0.33573234, 0.33332485])
        # q = np.array([0.3345275, 0.33213279, 0.33333971])
        # tmp, minV_1, maxV_1, minV_2, maxV_2, vly = calc_eps(utility, p, q)
        # print(tmp, minV_1, maxV_1, minV_2, maxV_2, vly)
        # exit()

    

        # count1 = np.zeros(n_action[1])
        # count1[0] = 1
        # count2 = np.zeros(n_action[0])
        # count2[1] = 1
        
        # run_FPs(n_action, utility, count1=count1, count2=count2, ratio=ratio, T=int(1e4))
        # exit()

    x = []
    values = []
    for ratio in ratios:
        # ratio = k * 1.0 / acc
        # k = 999
        # k = 5000
        # ratio = k * 1.0 / 10000
        # ratio = 0.005
        
        print('Potential\n', payoff1)
        print('Harmonic\n', payoff2)
        
        print('Ratio %.4f P %.4f H' % (ratio, 1 - ratio))
        utility = ratio * payoff1 + (1 - ratio) * payoff2
        print('Utility\n', utility)
        
        
        # utility = utility_normalize(utility)
        

        ans = run_findRC(n_action, utility, T=T_RC)
        print('tag', ans[0])

        if ans[0] is False:
            break

        x.append(ratio)
        values.append(np.array(ans).flatten())
        # exit()

    x = np.array(x).reshape((-1, 1))
    values = np.array(values)
    print('x', x, x.shape)
    print('val', values, values.shape)

    datas = np.hstack((x, values))
    # np.save('datas_rand_1.npy', np.array(datas))
    
    print('datas', datas.shape)

    plt.plot(np.array(datas)[:, 0], np.array(
        datas)[:, 1], label='maxV')
    plt.legend()
    
    plt.figure()
    plt.plot(np.array(datas)[:, 0], np.array(
        values)[:, 3], label='minV')
    
    plt.legend()
    # plt.show()


def run_matrix_construct():
    payoff, n_agent, n_action = input_game(n_action=[3, 3])
    ldpayoff = payoff2ld(payoff)
    gd = GameDecompostion(n_agent=n_agent, n_action=n_action, payoff=ldpayoff)
    potential, harmonic, nonstra = gd.main()

    # potential[1, :, :] = potential[0, :, :]

    # acc = 10
    # for k in range(acc + 1):
    #     ratio = k * 1.0 / acc
    #     print('Ratio %.4f P %.4f H' % (ratio, 1 - ratio))
    #     utility = ratio * potential + (1 - ratio) * harmonic
    #     # utility = utility_normalize(utility)

    #     print('Utility', utility)
    #     print('Potential', ratio * potential)
    #     print('Harmonic', (1 - ratio) * harmonic)
    #     run_findRC(n_action, utility, T=int(1e7))
        
        # p = np.array([0.57336197, 0.30310057, 0.12353746])
        # q = np.array([0.24693794, 0.34533383, 0.40772823])
        # p = np.array([0.33094281, 0.33573234, 0.33332485])
        # q = np.array([0.3345275, 0.33213279, 0.33333971])
        # tmp, minV_1, maxV_1, minV_2, maxV_2, vly = calc_eps(utility, p, q)
        # print(tmp, minV_1, maxV_1, minV_2, maxV_2, vly)
        # exit()

    

        # count1 = np.zeros(n_action[1])
        # count1[0] = 1
        # count2 = np.zeros(n_action[0])
        # count2[1] = 1
        
        # run_FPs(n_action, utility, count1=count1, count2=count2, ratio=ratio, T=int(1e4))
        # exit()

    for k in range(0, 1000, 100):
        # ratio = k * 1.0 / acc
        # k = 0
        k = 500
        ratio = k * 1.0 / 1000

        # ratio = 0.005
        print('Ratio %.4f P %.4f H' % (ratio, 1 - ratio))
        utility = ratio * potential + (1 - ratio) * harmonic
        # utility = utility_normalize(utility)

        print('Utility', utility)
        print('Potential', ratio * potential)
        print('Harmonic', (1 - ratio) * harmonic)
        # run_findRC(n_action, utility, T=int(1e5))
        
        RC = RobustCycle(n_action=n_action, payoff=utility)
        # rc = [((1, 0), (None, None)), 
        #       ((2, 0), (None, None)), 
        #       ((0, 0), (None, None)),
        #       ((0, 2), (None, None)),
        #       ((0, 1), (None, None)),
        #       ((2, 1), (None, None)),
        #       ((2, 2), (None, None)),
        #       ((1, 2), (None, None)),
        # ]

        # rc = [
        #     ((1, 0), (array([0.331681, 0.33234261, 0.33597639]),
        #               array([0.33229172, 0.33563031, 0.33207797])))
        #     ((2, 0), (array([0.33100044, 0.33371256, 0.33528701]), array(
        #         [0.33366177, 0.33494164, 0.33139659])))
        #     ((0, 0), (array([0.33002491, 0.33272904, 0.33724605]), array(
        #         [0.3356256, 0.33395451, 0.3304199])))
        #     ((0, 2), (array([0.33128462, 0.33210343, 0.33661194]), array(
        #         [0.33687477, 0.33332659, 0.32979863])))
        #     ((0, 1), (array([0.33313844, 0.33118277, 0.33567879]), array(
        #         [0.33594089, 0.33240254, 0.33165657])))
        #     ((2, 1), (array([0.33458734, 0.33046321, 0.33494945]), array(
        #         [0.33521098, 0.33385304, 0.33093598])))
        #     ((1, 1), (array([0.33331997, 0.32921147, 0.33746856]), array(
        #         [0.33394126, 0.3363763, 0.32968244])))
        #     ((1, 2), (array([0.33301296, 0.3298293, 0.33715773]), array(
        #         [0.33363368, 0.33698754, 0.32937879])))
        # ]
        tag, rc = RC.find_RC(int(1e6))
        
        for i in range(len(rc)):
            print(rc[i])

        RC.constructCD(rc)
        C, D = RC.constructCD(rc)
        print('C', C)
        print('D', D)
        M = np.linalg.inv(C) @ D
        w, v = np.linalg.eig(M)
        print('w', w, w.shape)
        print('v', v, v.shape)
        ansW = 1
        for b in w[1:]:
            ansW *= b
        print('ansW', ansW)

def figure_plot():
    # x = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54]
    # maxV = [0.2189, 0.2184, 0.2176, 0.2163, 0.2148, 0.2129, 0.2107, 0.2082, 0.2054, 0.2022]
    # minV = [0.0631, 0.0636, 0.0639, 0.0642, 0.0644, 0.0646, 0.0647, 0.0647, 0.0646, 0.0645]
    # x = np.load()
    # plt.plot(x, maxV, label='maxV')
    # plt.plot(x, minV, label='minV')
    plt.legend()
    plt.show()

# figure_plot()

if __name__ == '__main__':
    T = int(1e5)
    # def input_game(theta=5, n_action=[3, 3]):
    #     # A = np.array([
    #     #     [[2, -3, 1],
    #     #      [-2, 4, 1.5],
    #     #      [-1, -1.4, 1.7]],
    #     #     [[-2, 3, -0.5],
    #     #      [1, -3, 0.5],
    #     #      [0.2, 0.5, -2]]
    #     # ], dtype=float)
        
    #     # A = np.array([
    #     #     [[2, -3, 0],
    #     #      [-2, 4, 0],
    #     #      [0, 0, 0]],
    #     #     [[-2, 3, 0],
    #     #      [1, -3, 0],
    #     #      [0, 0, 0]]
    #     # ], dtype=float)
    #     # return A, 2, [3, 3]

    #     # A = np.array([
    #     #     [[0, 1, theta],
    #     #      [theta, 0, 1],
    #     #      [1, theta, 0]],
    #     #     [[0, theta, 1],
    #     #      [1, 0, theta],
    #     #      [theta, 1, 0]]
    #     # ], dtype=float)
    #     # return A, 2, [3, 3]

    #     A = np.random.rand(2, n_action[0], n_action[1])
    #     return A, 2, n_action
    
        # A = np.random.randn(2, n_action[0], n_action[1])
        # A[1, :, :] = -A[0, :, :]
        # return A, 2, n_action
        
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
        else:
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
            A = np.array([
            [[0, 2, 1],
             [1, 0, 2],
             [2, 1, 0]],
            [[0, 1, 2],
             [2, 0, 1],
             [1, 2, 0]]
            ], dtype=float)
            return A, 2, [3, 3]

        return A, 2, n_action

    payoff, n_agent, n_action = input_game(n_action=[3, 3], mode='other')
    ldpayoff = payoff2ld(payoff)
    gd = GameDecompostion(n_agent=n_agent, n_action=n_action, payoff=ldpayoff)
    potential, harmonic, nonstra = gd.main()
    
    # potential[1, :, :] = potential[0, :, :]
    # potential = normalized_game(potential)

    # print('Potential', potential)
    # print('Hamonic', harmonic)
    # print('nonstr', nonstra)
    # print('P+S', potential + nonstra)

    # exit()
    n_action = [5, 5]
    # ZS, _, _ = input_game(n_action=n_action, mode='ZS')
    
    # II1, _, _ = input_game(n_action=n_action, mode='II')
    # II2, _, _ = input_game(n_action=n_action, mode='II')

    HA, _, _ = input_game(n_action=n_action, mode='HA')
    NI, _, _ = input_game(n_action=n_action, mode='NI')
    OT, _, _ = input_game(n_action=n_action, mode='other')
    
    # A = np.array(
    #     [[ 0.33362501, -0.0256309,  -0.30799411],
    #      [-0.13503309, -0.08784788,  0.22288098],
    #      [-0.19859192,  0.11347878,  0.08511313]]
    # )   

    # HA = np.zeros((2, n_action[0], n_action[1]))
    # HA[0, :, :] = A
    # HA[1, :, :] = -A
    # print('NI + HA')

    # ratios = [0.0, 0.02, 0.04, 0.08, 0.16, 0.24]
    # ratios = np.arange(1, 100) / 100
    # run_matrix_method(n_action, potential * 0.1, harmonic, ratios, T_RC=int(1e7))

    # run_matrix_method(n_action, NI * 0.1, OT, ratios, T_RC=int(1e7))
    
    print('NI + HA')
    ratios = np.arange(0, 50) / 100
    ratios = [0.0]
    print('n_action', n_action)
    print('NI\n', NI)
    print('HA\n', HA)
    run_BRC_method(n_action, NI * 0.1, HA, ratios, T_RC=int(1e8))
    # run_matrix_method(n_action, NI * 0.1, HA, ratios, T_RC=int(1e7))

# run_matrix_construct()


