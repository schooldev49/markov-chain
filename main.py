
# imports

import networkx as nx
import numpy as np
import pandas as pd #type: ignore


# creating graph/nodes

G = nx.Graph()

G.add_nodes_from(['A','B','C','D','E'])
G.add_edges_from([('A','D'),('D','E'), ('A','B'), ('A','C'), ('C', 'D'), ('B','E'), ('A','E')])



# creating adjacency matrix

def createAdjacencyMatrix(nodes, edges):
    names = G.nodes 
    size = len(names)
    arr = np.zeros(shape=(size,size))
    matrix = pd.DataFrame(arr, index=names, columns=names)

    for edge in edges:
        start, end = edge
        matrix.loc[start][end] = 1
        matrix.loc[end][start] = 1

    return matrix 

matrix = createAdjacencyMatrix(G.nodes, G.edges)
print("ADJACENCY MATRIX...")
print(matrix)

# creating probability matrix

def createProbabilityMatrix(adjMatrix):

    for index, row in adjMatrix.iterrows():
        tempSum = row.sum()
        if (tempSum > 0):
            adjMatrix.loc[index] = (row / tempSum)
    
    return adjMatrix


def marchovChain(probMatrix, k):

    nodes = probMatrix.index.tolist()

    probZero = np.zeros(len(nodes))
    probZero[nodes.index('A')] = 1

    matrixTranspose = probMatrix.T.values
    
    tranpose_power = np.linalg.matrix_power(matrixTranspose, k)

    pk = np.dot(probZero, tranpose_power)

    return pd.Series(pk, index=nodes)

newMatrix = createProbabilityMatrix(matrix)


for i in range(1,27,4):
    print("After " + str(i) + " Steps:")
    k_step_matrix = marchovChain(newMatrix, i)
    print(k_step_matrix)



