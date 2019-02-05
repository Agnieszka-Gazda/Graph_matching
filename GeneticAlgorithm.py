import networkx as nx
import random
import math
import operator
import copy
import numpy as np
import time

start_time = time.time()

def check_if_ismorphic(G1, G2):
    '''
    First stage of algorithm, compare the parameters of two graphs in order to chceck if are isomorphic
    '''
    para1 = count_parameters(G1)
    para2 = count_parameters(G2)
    if len(para1) != len(para2):
        # different number of nodes
        return False
    else:
        for i, node in enumerate(para1):
            if node[1] != para2[i][1] or node[2] != para2[i][2] or node[3] != para2[i][3]:
                return False
    return True

def count_parameters(G):

    graph_parameters = []
    N = nx.number_of_nodes(G)
    # print(N)
    nodes = nx.nodes(G)
    for i in nodes:
        deg = nx.degree(G, i)
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

class GA:

    def __init__(self, G1, G2, pm=0.2, pcr=0.7, iter=5000, population=250, r0=0.01):
        self.G1 = G1
        self.G2 = G2
        self.ro = r0
        self.para1 = count_parameters(G1)
        self.para2=count_parameters(G2)
        self.iter=iter
        self.Init=population
        self.mutationProb=pm
        self.crossoverProb=pcr


    def create_initial_population(self):

        # print(Init)
        populationCR = []
        for i in range(1, int(self.Init)):
            populationCR.append(self.create_chromosome_random())

        populationCR_fitness = []
        for chromosome in populationCR:
            fitness = self.fitness_function(chromosome)
            temp = dict()
            temp['chromosome'] = chromosome
            temp['fitness'] = fitness
            populationCR_fitness.append(temp)
        # print(populationCR_fitness)

        return populationCR_fitness

    def create_chromosome_random(self):
        '''
        gen = (v1, d1, e1,)
        one chromosome is one whole fit
        '''
        para2=copy.deepcopy(self.para2)
        para1=copy.deepcopy(self.para1)
        random.shuffle(para2)
        chromosome = []
        for i, node in enumerate(para1):
            gen = node + para2[i]
            chromosome.append(gen)

        #print(chromosome)
        return chromosome

    def fitness_function(self, chromosome):
        sum = 0
        for gen in chromosome:
            inv = abs(gen[1] * gen[2] * gen[3] - gen[5] * gen[6] * gen[7])
            sum = sum + inv
        fitness = 1 / (sum + self.ro)
        # print('FITNESS: {}, {}, {}'.format(chromosome, fitness, ro))
        return fitness

    def pick_parents_ranking_list(self, population):
        population.sort(key=operator.itemgetter('fitness'))
        # print('sorted: {}'.format(population))
        parents = []
        parents.append(population[-1])
        parents.append(population[-2])
        return parents

    def pick_parents_roulette(self, population):
        # not tested
        sum=0;
        fitness_list=[]

        chromosome_list=[]
        for i in population:
            sum=sum+i['fitness']
            fitness_list.append(i['fitness'])
            chromosome_list.append(str(i['chromosome']))

        size=len(fitness_list)

        for i in range(0, size):
            fitness_list[i]=fitness_list[i]/sum
        parents=[]
        parents.append(np.random.choice(chromosome_list, 1, p=fitness_list))
        parents.append(np.random.choice(chromosome_list, 1, p=fitness_list))
        #print(parents)
        return parents



    def crossover(self, parents):

        parent0 = parents[0]['chromosome']
        parent1 = parents[1]['chromosome']

        n = len(parents[0]['chromosome'])
        CP1 = random.randint(0, n - 1)
        CP2 = random.randint(0, n - 1)
        #print(CP1, CP2)

        temp = copy.deepcopy(parent1[CP2])
        temp2 = copy.deepcopy(parent0[CP1])
        in_first = parents[0]['chromosome'][CP1]
        in_second = parents[1]['chromosome'][CP2]


        for i in parents[0]['chromosome']:
            changed_in_first = in_second[0]
            if i[0] == changed_in_first:
                i[0:4] = in_first[0:4]
            if i[4] == in_second[4]:
                i[4:8] = in_first[4:8]

        for i in parents[1]['chromosome']:
            changed_in_second = in_first[0]
            if i[0] == changed_in_second:
                i[0:4] = in_second[0:4]
            if i[4] == in_first[4]:
                i[4:8] = in_second[4:8]

        parents[0]['chromosome'][CP1] = temp
        parents[1]['chromosome'][CP2] = temp2

        parents[0]['fitness'] = self.fitness_function(parents[0]['chromosome'])
        parents[1]['fitness'] = self.fitness_function(parents[1]['chromosome'])
        return parents

    def mutation(self, parents):
        n = len(parents[0]['chromosome'])
        CP1 = random.randint(0, n - 1)
        CP2 = random.randint(0, n - 1)

        # first child
        temp = copy.deepcopy(parents[0]['chromosome'][CP1])
        parents[0]['chromosome'][CP1][0:4] = parents[0]['chromosome'][CP2][0:4]
        parents[0]['chromosome'][CP2][0:4] = temp[0:4]

        parents[0]['fitness'] = self.fitness_function(parents[0]['chromosome'])
        parents[1]['fitness'] = self.fitness_function(parents[1]['chromosome'])

        return parents

    def Generic_algorithm(self):
        n = nx.number_of_nodes(self.G1)

        if n <= 5:
            Init = math.factorial(n) / 5
        elif n > 5 and n <= 7:
            Init = math.factorial(n) / 20
        Init=copy.deepcopy(self.Init)

        population = self.create_initial_population()
        #parents = self.pick_parents_ranking_list(population)
        parents = self.pick_parents_roulette(population)


        for i in range(0, self.iter):
            parents = self.pick_parents_ranking_list(population)
            M = parents[0]['fitness']
            f = open('output.txt', 'w')
            if (M == 100):
                f.write("GA\n")
                f.writelines("Found in {} iteration\n".format(i))
                f.writelines("{" + "{}, fitness: {} ". format(parents[0]['chromosome'], parents[0]['fitness']) +"}")
                f.close
                print("Found in {} iteration".format(i))
                print("{" + "{}, fitness: {} ". format(parents[0]['chromosome'], parents[0]['fitness']) +"}")
                break

            chrprob=random.random()
            if chrprob<self.crossoverProb:
                parents = copy.deepcopy(self.crossover(parents))

            mutprob = random.random()
            if mutprob<self.mutationProb:
                parents = copy.deepcopy(self.mutation(parents))
            # print(muted_child[0]['fitness'])
            # print(muted_child[1]['fitness'])
            population.sort(key=operator.itemgetter('fitness'))
            if (population[-1]['fitness']) < parents[0]['fitness']:
                population[-1] = copy.deepcopy(parents[0])
            if (population[-2]['fitness']) < parents[0]['fitness']:
                population[-2] = copy.deepcopy(parents[0])
            if (population[-1]['fitness']) < parents[1]['fitness']:
                population[-1] = copy.deepcopy(parents[1])
            if (population[-2]['fitness']) < parents[1]['fitness']:
                population[-2] = copy.deepcopy(parents[1])
            #print(M)

        #print(parents[0])

'''
G = nx.Graph()

G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9 ,10])
G.add_edges_from([(1, 3), (1, 4), (4, 3), (4, 2), (2, 5), (5, 1), (1, 10), (10, 9), (9, 2), (8, 2), (7, 4), (8, 6)])

G2 = nx.Graph()

G2.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9 ,10])
G2.add_edges_from([(1, 3), (1, 4), (4, 3), (4, 2), (2, 5), (5, 1), (1, 10), (10, 9), (9, 2), (8, 2), (7, 4), (8, 6)])


GA=GA(G, G2)
GA.Generic_algorithm()
elapsed_time = time.time() - start_time
print(elapsed_time)
'''