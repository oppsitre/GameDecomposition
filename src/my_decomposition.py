import numpy as np
import copy
from gen_game import *

class GameDecompostion:
    def __init__(self, n_action, n_agent, payoff):
        self.n_action = n_action
        self.n_agent = n_agent
        self.payoff = payoff
        # self.payoff = payoff

        # construct node to ID
        self.E2ID = {}
        self.ID2E = {}
        self.eid = 0

        def con_E_ID(aid, acts, eid):
            if aid == self.n_agent:
                self.E2ID[tuple(acts)] = self.eid
                self.ID2E[self.eid] = tuple(acts)
                self.eid += 1
                return

            for a in range(n_action[aid]):
                acts.append(a)
                con_E_ID(aid + 1, acts, eid)
                acts.pop()

        con_E_ID(0, [], 0)

        print(self.E2ID)
        # exit()

        # print(self.ID2E)

        self.n_node = len(self.E2ID.keys())
        self.n_edge = self.n_node * self.n_node

        def N2E(p, q):
            return p * self.n_node + q

        self.Dm = np.zeros((self.n_agent, self.n_edge, self.n_node))
        
        for i in range(self.n_agent):
            for p in range(self.n_node):
                for q in range(self.n_node):
                    if self.is_m_comparable(p, q, i):
                        self.Dm[i, N2E(p, q), q] = 1
                        self.Dm[i, N2E(p, q), p] = -1

        self.delta_0 = np.sum(self.Dm, axis=0)

        self.Um = np.zeros((self.n_agent, self.n_node))
        for i in range(self.n_agent):
            for key in self.payoff[i].keys():
                eid = self.E2ID[key]
                self.Um[i, eid] = self.payoff[i][key]

        self.Pim = np.zeros((self.n_agent, self.n_node, self.n_node))
        for i in range(self.n_agent):
            self.Pim[i, :] = np.matmul(np.linalg.pinv(self.Dm[i, :]), self.Dm[i, :])

        self.X = np.zeros(self.n_edge)
        for i in range(self.n_agent):
            self.X += np.matmul(self.Dm[i, :], self.Um[i, :].T)

        self.phi = np.matmul(np.linalg.pinv(self.delta_0), self.X)

        self.pot = np.zeros((self.n_agent, self.n_node))
        self.har = np.zeros((self.n_agent, self.n_node))
        self.non = np.zeros((self.n_agent, self.n_node))
        for i in range(n_agent):
            self.pot[i, :] = np.matmul(self.Pim[i, :], self.phi)

        for i in range(n_agent):
            self.non[i, :] = self.Um[i, :] - np.matmul(self.Pim[i, :], self.Um[i, :].T)

        self.har = self.Um - self.pot - self.non


    # def normalize(self, payoff):
    #     for i in range(self.n_agent):
    #         div = np.max(np.abs(payoff[i]))
    #         payoff[i] = payoff[i] / div
    #     return payoff

    def operator_inner_product(self, x, y):

        return np.sum(x * y)

    def calc_alpha(self, G1, G2):
        ans = 0
        for i in range(self.n_agent):
            dif = G1[i, :] - G2[i, :]
            inn = self.operator_inner_product(dif, dif)
            ans += inn * self.n_action[i]

        return np.sqrt(ans)
        # for e in range(self.n_node):

    def is_m_comparable(self, p, q, m):
        ep = self.ID2E[p]
        eq = self.ID2E[q]

        for i in range(len(ep)):
            if i != m and ep[i] != eq[i]:
                return False
            if i == m and ep[i] == eq[i]:
                return False
        return True


    def main(self):
        # print('potential', self.pot)
        # print('harmonic', self.har)
        # print('non-strategy', self.non)
        # print('alpha', self.calc_alpha(self.Um, self.pot))
        self.dis_pot = self.calc_alpha(self.Um, self.pot)
        self.dis_har = self.calc_alpha(self.Um, self.har)
        
        res_pot = []
        res_har = []
        res_non = []
        for i in range(self.n_agent):
            pot = np.expand_dims(np.reshape(self.pot[i, :], tuple(self.n_action), order='C'), axis=0)
            har = np.expand_dims(np.reshape(
                self.har[i, :], tuple(self.n_action), order='C'), axis=0)
            non = np.expand_dims(np.reshape(self.non[i, :], tuple(self.n_action), order='C'), axis=0)
            # print('pot', pot)
            # print('har', har)
            # print('non', non)

            res_pot.append(pot)
            res_har.append(har)
            res_non.append(non)

        # print('adfa')
        # print(np.vstack(res_pot))
        # print(np.vstack(res_har))
        # print(np.vstack(res_non))
        return np.vstack(res_pot), np.vstack(res_har), np.vstack(res_non)
        # return self.pot, self.har, self.non

# def input_game():
#     A = np.array([
#         [[3, 0],
#          [5, 1]],
#         [[3, 5],
#          [0, 1]],
#     ])

#     return A, 2, [2, 2]


def input_game():
    A = np.array([
        [[0, 2, 1],
         [1, 0, 2],
         [2, 1, 0]],
        [[0, 1, 2],
         [2, 0, 1],
         [1, 2, 0]]
    ])

    return A, 2, [2, 2]

# def input_game():
#     A = np.array([
#         [[3, -2],
#          [-2, 1]],
#         [[-3, 2],
#          [2, -1]]
#     ])

#     return A, 2, [2, 2]


# def input_1(theta=0.1):
#     A = np.array([
#         [[0, 1, theta / 2.0],
#          [1, 0, 0],
#          [1, 0, theta]],
#         [[1, 0, 0],
#          [0, 1, 1],
#          [0, theta / 2.0, theta]]

#     ])
#     return A, 2, [3, 3]


# def input_2(theta=0.1):
#     A = np.array([
#         [[0, 1, theta / 2.0],
#          [1, 0, 0],
#          [1, 0, -theta]],
#         [[1, 0, 0],
#          [0, 1, 1],
#          [0, theta / 2.0, -theta]]
#     ])
#     return A, 2, [3, 3]

# def input_game(e=0.25):
#     A = np.array([
#         [[0, -1, -e],
#          [1, 0, -e],
#          [e, e , 0]],
#         [[0, 1, e],
#          [-1, 0, e],
#          [-e, -e, 0]],
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

#     # A[0, :, :] -= np.ones((3, 3), dtype=float)
#     # A[1, :, :] -= np.ones((3, 3), dtype=float)
#     # print('A', A)
#     return A, 2, [3, 3]

# def input_game():
#     A = np.array([
#         [[0, 1, -1],
#          [-1, 0, 1],
#          [1, -1, 0]],
#         [[0, -1, 1],
#          [1, 0, -1],
#          [-1, 1, 0]]
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



def denoise(P):
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            for k in range(P.shape[2]):
                if abs(P[i, j, k]) < 1e-10:
                    P[i, j, k] = 0
    return P

# def construct_Harmonic(n_action = [3, 3]):
    # A = np.zeros(2, n_action[0], n_action[1])
    # for i in range(n_action[0] - 1):
    #     for j in range(n_action[1] - 1):
    #         A[0, i, j] = np.random.randn()
    #     A[0, i, -1] = -np.sum(A[0, :, :])
    # for j in range()
    # A[0, -1, j]
    # A[1, :, :] = -A[0, :, :]


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
    elif mode == 'HA+II':
        payoff = np.random.rand(2, n_action[0], n_action[1])
        ldpayoff = payoff2ld(payoff)
        gd = GameDecompostion(
            n_agent=2, n_action=n_action, payoff=ldpayoff)
        potential, harmonic, nonstra = gd.main()

        C = np.random.rand(2, n_action[0], n_action[1])
        C[1, :, :] = C[0, :, :]

        A = harmonic + C

    elif mode == 'ZS':
        A = np.random.rand(2, n_action[0], n_action[1])
        A[1, :, :] = -A[0, :, :]

    elif mode == 'II':
        A = np.random.rand(2, n_action[0], n_action[1])
        A[1, :, :] = A[0, :, :]
        # A = utility_normalize(A)

    elif mode == 'HA':
        payoff = np.random.rand(2, n_action[0], n_action[1])
        ldpayoff = payoff2ld(payoff)
        gd = GameDecompostion(
            n_agent=2, n_action=n_action, payoff=ldpayoff)
        potential, harmonic, nonstra = gd.main()

        A = harmonic

    elif mode == 'random':
        A = np.random.rand(2, n_action[0], n_action[1])

    else:
        A = np.array([
            [[0, 2, 1],
            [1, 0, 2],
            [2, 1, 0]],
            [[0, 1, 2],
            [2, 0, 1],
            [1, 2, 0]]
        ])

        return A, 2, [3, 3]
        # A = np.array(
        #     [
        #         [[3, -3, 0],
        #          [-3, 3, 0]],
        #         [[-2, 2, 0],
        #          [2, -2, 0]]
        #     ]
        # )
        # return A, 2, [2, 3]

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
            #     [[4, -1, 1],
            #  [1, 2, -2],
            #  [-1, 0, 2]],
            #     [[4, 1, -1],
            #  [-1, 2, 0],
            #  [1, -2, 2]]
            # ], dtype=float)
            # return A, 2, [3, 3]

    return A, 2, n_action

if __name__ == '__main__':
    payoff, n_agent, n_action = input_game(n_action=[3, 3], mode='other')
    # payoff, n_agent, n_action = input_game(e=1e-9)
    print('n_action', n_action, payoff.shape)
    ldpayoff = payoff2ld(payoff)
    gd = GameDecompostion(n_agent=n_agent, n_action=n_action, payoff=ldpayoff)
    potential, harmonic, nonstra = gd.main()
    print('potential\n', denoise(potential), np.sum(
        potential[0, :, :], axis=0), np.sum(potential[1, :, :], axis=1))

    print('harmonic\n', denoise(harmonic))
    print('nonstra\n', denoise(nonstra))

    ans = np.dot(potential, harmonic)

    print('P * H', np.sum(ans))
    print('P + S\n', potential + nonstra)
    
    exit()

    print('Decomposition of Harmonic Game')
    ldharmonic = payoff2ld(harmonic)
    gd = GameDecompostion(n_agent=n_agent, n_action=n_action, payoff=ldharmonic)
    potential, harmonic, nonstra = gd.main()
    print('potential\n', potential)
    print('harmonic\n', harmonic)
    print('nonstra', nonstra)

    # payoff, n_agent, n_action = input_2()
    # ldpayoff = payoff2ld(payoff)
    # gd = GameDecompostion(n_agent=n_agent, n_action=n_action, payoff=ldpayoff)
    # potential, harmonic, nonstra = gd.main()
    # print('potential\n', potential)
    # print('harmonic\n', harmonic)
    # print('nonstra', nonstra)
