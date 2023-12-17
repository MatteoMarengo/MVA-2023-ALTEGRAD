"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
##################

# Load the data into an undirected graph G
G = nx.read_edgelist('D:\OneDrive\Documents\MVA-ENS-2023-2024\S1\LEARNING\AlteGrad\lab2_graph_mining\code\datasets\CA-HepTh.txt', delimiter='\t', comments='#', create_using=nx.Graph())

# Compute network characteristics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Print the characteristics
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)





############## Task 2

##################
# your code here #
##################



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################



############## Task 4

##################
# your code here #
##################




############## Task 5

##################
# your code here #
##################
