from ACO import *
import GeneticAlgorithm
import networkx as nx
import configparser



config = configparser.ConfigParser()
config.read('input.ini')
nodes_G1=config['GRAPHS']['nodes_G1']
nodes_G2=config['GRAPHS']['nodes_G2']

edges_G1=config['GRAPHS']['edges_G1'].split('\n')
edges_G2=config['GRAPHS']['edges_G2'].split('\n')

mode=config['MODE']['algorithm']
print(mode)

#ANT ALGO PARAMETERS
ANT_config_alpha=config['ANT']['alpha']
ANT_config_beta=config['ANT']['beta']
ANT_config_ant=config['ANT']['ant']
ANT_config_gen=config['ANT']['gen']

#GA PARAMETERS
GA_config_mut=config['GA']['pr_mut']
GA_config_cr=config['GA']['pr_cr']
GA_config_iter=config['GA']['iter']
GA_config_pop=config['GA']['populacja']



for i, edge in enumerate(edges_G1):
    edges_G1[i]=eval(edge)

for i, edge in enumerate(edges_G2):
    edges_G2[i] = eval(edge)

nodes_G1=eval(nodes_G1)
nodes_G2=eval(nodes_G2)

G1 = nx.Graph()
G1.add_nodes_from(nodes_G1)
G1.add_edges_from(edges_G1)

G2 = nx.Graph()
G2.add_nodes_from(nodes_G2)
G2.add_edges_from(edges_G2)
constGraph=nx.Graph()


if mode=='ANT':
    A=ACO(G1, G2, int(ANT_config_ant), int(ANT_config_gen), float(ANT_config_alpha), float(ANT_config_beta))
    A.ACO_algo()


if mode=='GA':
    ga=GeneticAlgorithm.GA(G1, G2, float(GA_config_mut), float(GA_config_cr), int(GA_config_iter), int(GA_config_pop))
    ga.Generic_algorithm()

