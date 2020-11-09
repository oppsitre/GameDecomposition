import numpy as np
import copy
from gen_game import *
from my_decomposition import GameDecompostion


class MCC:
    def __init__(self, payoff=None, n_agent=None, n_action=None):
        
        # dfn: 结点的时间戳，顺便充当标记数组，0代表未被访问，大于0表示访问过
        # low: dfs树的根，强连通分量的根
        self.n_agent = n_agent
        self.n_action = n_action
        self.payoff = payoff

        n_node = 1 
        for x in n_action:
            n_node *= x
        self.n_node = n_node
        self.dfn = [0] * n_node
        self.low = [0] * n_node
        
        self.stack = []
        # 标记是否在栈中
        self.in_stack = [False] * n_node

        # 当前时间，时间戳
        self.idx = 0

        # 图中包含的所有的强连通分量
        self.sccs = {}

        # 强连通分量的数量
        self.bcnt = 0

        # if payoff is not None:
        #     self.graph = self.construct_graph(payoff)
    
    # def construct_graph(self, payoff):

    #     # 数据
    #     # edges = [
    #     #     [0, 1],
    #     #     [1, 2],
    #     #     [2, 3],
    #     #     [2, 4],
    #     #     [3, 4],
    #     #     [3, 1],
    #     #     [0, 5],
    #     #     [5, 6],
    #     #     [6, 0],
    #     # ]
    #     # n = 7


    #     # 构图
    #     graph = [[] for _ in range(self.n_node)]
    #     for a, b in edges:
    #         graph[a].append(b)
    #     return graph
        

    def tarjan(self, u):
        self.idx += 1
        self.dfn[u] = copy.deepcopy(self.idx)
        self.low[u] = copy.deepcopy(self.idx)
        self.stack.append(u)
        self.in_stack[u] = True

        for v in self.graph[u]:
            if self.dfn[v] == 0:
                self.tarjan(v)  # dfs(v)
                self.low[u] = min(self.low[u], self.low[v])  # 回溯
            elif self.in_stack[v] is True:
                self.low[u] = min(self.low[u], self.dfn[v])

        # 回溯后，判断dfn[u] == low[u]，把u以上的点弹出
        if self.dfn[u] == self.low[u]:
            self.sccs[self.bcnt] = []
            while True:
                v = self.stack.pop()
                self.in_stack[v] = False
                self.sccs[self.bcnt].append(v)
                if u == v:
                    break
            self.bcnt += 1

    def find_Mcc(self, graph):
        return

    def main(self):
        self.tarjan(0)
        print('sccs', self.sccs)
        # for k, v in self.sccs.items():
        #     print(k, [chr(e + ord('a')) for e in sorted(v)])
        return self.n_agent

    def matrix2graph(self):
        ID2E = {}
        E2ID = {}
        self.eid = 0
        def con_E_ID(aid, acts):
            if aid == self.n_agent:
                E2ID[tuple(acts)] = self.eid
                ID2E[self.eid] = tuple(acts)
                self.eid += 1
                return

            for a in range(self.n_action[aid]):
                acts.append(a)
                con_E_ID(aid+1, acts)
                acts.pop()
        
        con_E_ID(0, [])
        # print('ID2E', ID2E)

        def is_m_improveable(p, q, m):
            ep = ID2E[p]
            eq = ID2E[q]

            for i in range(len(ep)):
                if i != m and ep[i] != eq[i]:
                    return False
                if i == m and ep[i] == eq[i]:
                    return False

            # for i in range(self.n_agent):
            aep = [m]
            aep.extend(ep)
            aep = tuple(aep)

            aeq = [m]
            aeq.extend(eq)
            aeq = tuple(aeq)

            if self.payoff[aeq] > self.payoff[aep]:
                # print('aeq', aeq, self.payoff[aeq])
                # print('aep', aep, self.payoff[aep])
                return True

            return False
        
        n_node = 1
        for x in self.n_action:
            n_node *= x

        # W = np.zeros((n_agent, n_node, n_node))
        graph = [[] for _ in range(n_node)]
        for p in range(n_node):
            for q in range(n_node):
                if q not in graph[p]:
                    for i in range(self.n_agent):
                        if is_m_improveable(p, q, i) is True:
                            # print('p', p, ID2E[p])
                            # print('q', q, ID2E[q])
                            graph[p].append(q)

        # print('graph', graph)
        self.graph = graph
        self.ID2E = ID2E
        self.E2ID = E2ID
        # return graph, ID2E, E2ID
        
        #     W[i, p, q] = 0
        # else:
        #     W[i, p, q] = 1
        # W = np.sum(W, axis=0)
        # graph = []
        # for a in range():
        # return W, ID2E, E2ID


if __name__ == '__main__':
    # payoff, n_agent, n_action = input_game()
    # mcc = MCC(payoff=payoff, n_agent=n_agent, n_action=n_action)
    # mcc.matrix2graph()
    # exit()
    ldpayoff = None
    payoff = None

    for div in range(1, 10):
        k = 1.0 / div
        print('k', k)
        payoff, n_agent, n_action = input_game_3()
        # payoff, n_agent, n_action = read_pot_add_har(k=k)
    #     # print('payoff', payoff)
        
        # ldpayoff, n_agent, n_action = EPRS(x=0, y=0, z=3)
        # ldpayoff, n_agent, n_action = coingame()
        # ldpayoff, n_agent, n_action = random_game(n_agent=2, n_action=[5, 5])

        # payoff = ld2payoff(ldpayoff, n_agent=n_agent, n_action=n_action)
        # ldpayoff, n_player, n_action = random_game(n_agent=2, n_action=[5, 5])
        if ldpayoff is None:
            ldpayoff = payoff2ld(payoff)
        
        if payoff is None:
            payoff = ld2payoff(ldpayoff, n_agent, n_action)
        # print('payoff\n', payoff)
        mcc = MCC(payoff=payoff, n_agent=n_agent, n_action=n_action)
        # graph, ID2E, E2ID = 
        mcc.matrix2graph()
        # print('graph', graph)
        mcc.main()

        print(18, mcc.ID2E[18])
        print(14, mcc.ID2E[14])
        print(19, mcc.ID2E[19])

        gd = GameDecompostion(n_action, n_agent, ldpayoff)
        potential, harmonic, nonstrat = gd.main()
        # np.save('potential.npy', potential)
        # np.save('harmonic.npy', harmonic)
        # exit()

        # print('potential')
        # print(potential)
        print('potential distance', gd.dis_pot)
        mcc_pot = MCC(payoff=potential, n_agent=n_agent, n_action=n_action)
        # graph_pot, ID2E, E2ID = 
        mcc_pot.matrix2graph()
        mcc_pot.main()

        # print('harmonic')
        # print(harmonic)
        print('harmonic distance', gd.dis_har)
        mcc_har = MCC(payoff=harmonic, n_agent=n_agent, n_action=n_action)
        # graph_har, ID2E, E2ID = 
        mcc_har.matrix2graph()
        mcc_har.main()
        print()
        break
    

