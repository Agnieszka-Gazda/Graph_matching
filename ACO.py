import networkx as nx
import numpy as np
import copy
import time
#import matplotlib.pyplot as plt
import random

#start_time = time.time()


class ACO:

    constrGraph1=nx.Graph()
    seed=200

    def __init__(self, G1, G2, ant_count=1, generations=30, alpha=1, beta=1, rho=0.3, q=1):
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.G1=G1
        self.G2=G2

    def initate_constrGraph(self, G1, G2):
        G1_list=count_parameters(G1)
        G2_list=count_parameters(G2)

        allowed = []
        for i in G1_list:
            for j in G2_list:
                allowed.append((i, j))

        for i in allowed:
            ACO.constrGraph1.add_node(str(i))

        for node1 in ACO.constrGraph1.nodes:
            for node2 in ACO.constrGraph1.nodes:
                ACO.constrGraph1.add_edge(node1, node2, weigth=1)


    def update_pheromone(self, bestAnt, bestCurrentM):

        # decrease the level of pheromone
        for edg in ACO.constrGraph1.edges():
            edg1 = edg[0]
            edg2 = edg[1]
            try:
                ACO.constrGraph1[edg1][edg2]['weight'] = ACO.constrGraph1[edg1][edg2]['weight'] *(1 - self.Q)
            except:
                ACO.constrGraph1[edg1][edg2]['weight']=1

            pheromone = 1/(1+self.final_score(bestAnt)-self.final_score(bestCurrentM))

            size=len(bestCurrentM)
            # adding more pheromone egde between every node in currenty best matching
            for i in range(0, size-1):
                for j in range(1, size):
                    try:
                        ACO.constrGraph1[str(bestCurrentM[i])][str(bestCurrentM[j])]['weight'] = ACO.constrGraph1[str(bestCurrentM[i])][str(bestCurrentM[j])]['weight'] + pheromone

                    except:
                        ACO.constrGraph1[str(bestCurrentM[i])][str(bestCurrentM[j])]['weight']=pheromone


    def score(self, matching, ro = 0.01):

        inv = abs(matching[0][1] * matching[0][2] * matching[0][3] - matching[1][1] * matching[1][2] * matching[1][3])
        score = 1 / (inv + ro)
        return score

    def ACO_algo(self):

        iteration_nbr=1
        self.initate_constrGraph(self.G1, self.G2)
        best_matching=[]
        best_matching_score=0

        #creating ants
        ants=[]
        for i in range(0, self.ant_count):
            A=ANT(ACO.constrGraph1, self.G1, self.G2, self.alpha, self.beta)
            ants.append(A)

        for n in range(0, self.generations):
            best_current=[]
            best_current_score=0
            if not self.stop_condition(best_matching) or best_matching_score!=100:
                for A in ants:
                    while len(A.visited)<(nx.number_of_nodes(self.G2)+4) and not self.stop_condition(A.visited):
                        if A.allowed:
                            A.add_matching()
                        Ascore=self.final_score(A.visited)

                    if Ascore>best_current_score:
                        best_current_score=copy.deepcopy(Ascore)
                        best_current=copy.deepcopy(A.visited)

                    if Ascore>best_matching_score:
                        best_matching_score=copy.deepcopy(Ascore)
                        best_matching=copy.deepcopy(A.visited)

                    A.visited=[]
                    A.allowed=A.candidate(self.G1, self.G2)
                self.update_pheromone(best_current, best_matching)

                iteration_nbr=copy.deepcopy(iteration_nbr)+1

        #print output into output.txt
        f = open('output.txt', 'w')
        f.write('ANT\n')
        f.writelines("ZNALAZŁEM W {} ITERACAJACH\n".format(iteration_nbr))
        f.writelines("BEST MATCHING score {} {}".format(best_matching, self.final_score(best_matching)))
        f.close()

        #print final output
        print("WYNIK KONCOWY")
        print("ZNALAZŁEM W {} ITERACAJACH".format(iteration_nbr))
        print("BEST MATCHING {} {}".format(best_matching, self.final_score(best_matching)))
        return best_matching


    def stop_condition(self, bestmatching):
        '''
        checks if all of the nodes from G1 have matching from G2
        '''

        G1nodes = self.G1.nodes
        for node in G1nodes:
            flag=0
            for i in bestmatching:
                if i[0][0]==node:
                    flag=1

            if flag==0:
                return False
        return True


    def final_score(self, m, ro=0.01):
        inv=0
        for matching in m:
            inv = inv+ abs(matching[0][1] * matching[0][2] * matching[0][3] - matching[1][1] * matching[1][2] * matching[1][3])
        return 1/(inv+ro)

class ANT(ACO):

    def __init__(self, constrGraph, G1, G2, alpha, beta):
        self.allowed = self.candidate(G1, G2)
        self.visited = []
        self.visited_Graph=nx.Graph()
        self.pheromone_cand=0;
        self.current=[]
        self.current_cost=0
        self.ANTscore=0;
        self.alpha=alpha
        self.beta=beta

    def candidate(self, G1, G2):
        G1_list=count_parameters(G1)
        G2_list=count_parameters(G2)

        allowed = []
        for i in G1_list:
            for j in G2_list:
                allowed.append((i, j))

        return allowed

    def add_matching(self):
        #choice of the next matching

        sum=0
        cand_prob=[]

        #pick next matching with the calculated probabilty
        for cand in self.allowed:
            sum=sum+self.prob(cand)
            cand_prob.append(self.prob(cand))
        size=len(self.allowed)

        for i in range(0, size):
            cand_prob[i]=cand_prob[i]/sum

        cands=list(range(0, size))
        ACO.seed=ACO.seed+14
        np.random.seed(ACO.seed)
        if (cands):
            next=np.random.choice(cands, 1, p=cand_prob)
            next=self.allowed[next[0]]

            ACO.constrGraph1.add_node(str(next))
            self.visited.append(next)
            self.visited_Graph.add_node(str(next))

        #removes used candidate (maybe add removing the impossible ones to?)
            for cand in self.allowed:
                if cand==next:
                    self.allowed.remove(cand)

            self.ANTscore=super().final_score(self.visited)
            self.current_cost=super().score(next)

    def prob(self, cand):
        pher=0
        self.pheromone_cand=0
        n=len(self.visited)
        if self.visited:
            for i in range(0,n):
                try:
                    temp = ACO.constrGraph1[str(self.visited[i])][str(cand)]['weigth']
                except:
                    temp=0
                    ACO.constrGraph1.add_edge(str(self.visited[i]), str(cand), weigth=1)

                pher=pher+temp
        else:
            pher=1

        self.pheromone_cand = self.pheromone_cand + pher

        h=self.score(cand)
        #print(self.pheromone_cand, h)
        prob=pow(pher, self.alpha)*pow(h, self.beta)
        return prob

def count_parameters(G):
    '''
    :return: matrix with parameters to count fittness function
    [node, degree, sum of shortest paths, eccentrity (max shortest path)]
    '''
    graph_parameters = []
    N = nx.number_of_nodes(G)
    nodes = nx.nodes(G)
    for i in nodes:
        deg = nx.degree(G,i)
        # if Sp=0, then the SP doesn't exist
        SP = nx.single_source_shortest_path_length(G, i)
        sum_SP = 0
        Ecc = 0
        for j in SP:
            if SP[j] > Ecc:
                Ecc = SP[j]
            sum_SP = sum_SP + SP[j]
        param = [i, deg, sum_SP, Ecc]
        graph_parameters.append(param)

    return graph_parameters

'''
G = nx.Graph()

G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9 ,10])
G.add_edges_from([(1, 3), (1, 4), (4, 3), (4, 2), (2, 5), (5, 1), (1, 10), (10, 9), (9, 2), (8, 2), (7, 4), (8, 6)])

G2 = nx.Graph()

G2.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9 ,10])
G2.add_edges_from([(1, 3), (1, 4), (4, 3), (4, 2), (2, 5), (5, 1), (1, 10), (10, 9), (9, 2), (8, 2), (7, 4), (8, 6)])


#constGraph=nx.Graph()
#plt.subplot(121)
#nx.draw(G, with_labels=True, font_weight='bold')
#plt.show()

A=ACO(G, G2)
A.ACO_algo()

elapsed_time = time.time() - start_time
print(elapsed_time)
'''