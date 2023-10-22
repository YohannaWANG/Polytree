import networkx as nx
import numpy as np

import config
p = config.setup()
lgr = p.logger

import random

seed = 123
random.seed(seed)
np.random.seed(seed)

"""
p.n: number of nodes
p.s: number of samples
"""
#n, s = p.n, p.s
#n = 5    # number of nodes
#s = 100000  # number of samples

"""
Function: Synthetic data generation. Move it to data.py
"""
def ER(p):
    '''
    simulate Erdos-Renyi (ER) DAG through networkx package

    Arguments:
        p: argparse arguments

    Uses:
        p.n: number of nodes
        p.d: degree of graph
        p.rs: numpy.random.RandomState

    Returns:
        B: (n, n) numpy.ndarray binary adjacency of ER DAG
    '''
    n, d = p.n, p.d
    p = float(d)/(n-1)

    G = nx.generators.erdos_renyi_graph(n=n, p=p, seed=5)

    U = nx.to_numpy_matrix(G)

    B = np.tril(U, k=-1)
    return B

def RT(p):
    '''
    simulate Random Tree DAG through networkx package
    Arguments:
    Arguments:
        p: argparse arguments
    Uses:
        p.n: number of nodes
        p.s: number of samples
        p.rs: numpy.random.RandomState
    '''
    n, s = p.n, p.s
    G = nx.random_tree(n, seed=15)
    U = nx.to_numpy_array(G)
    B = np.tril(U, k=-1)

    A = np.zeros((n, n))
    root = np.random.randint(n)
    for i in range(n):
        for j in range(n):
            # i,j not edge, continue
            if U[i, j] == 0:
                continue;
            elif j in nx.shortest_path(G, root, i):
                A[j, i] = 1
            elif i in nx.shortest_path(G, root, j):
                A[i, j] = 1

    return B


def WDAG(B, p):
    """
    Generate weighted DAG
    """
    A = np.zeros(B.shape)
    s = 1 # s is the scalling factor
    R = ((-0.5 * s, -0.1 * s), (0.1 * s, 0.5 * s))
    rs = np.random.RandomState(100) # generate random stets based on a seed
    S = rs.randint(len(R), size=A.shape)

    for i, (l, h) in enumerate(R):
        U = rs.uniform(low=l, high=h, size=A.shape)
        A += B * (S == i) * U

    return A

def SEM(A, p):
    '''
    simulate samples from linear structural equation model (SEM) with specified type of noise.
    Arguments:
        A: (n, n) weighted adjacency matrix of DAG
        p: argparse arguments
    Uses:
        p.n: number of nodes
        p.s: number of samples
        p.rs (numpy.random.RandomState): Random number generator
        p.tn: type of noise, options: ev, nv, exp, gum
            ev: equal variance
            uv: unequal variance
            exp: exponential noise
            gum: gumbel noise
    Returns:
        numpy.ndarray: (s, n) data matrix.
    '''
    n, s = p.n, p.s
    r = np.random.RandomState(500)
    def _SEM(X, I):

        '''
        simulate samples from linear SEM for the i-th vertex
        Arguments:
            X (numpy.ndarray): (s, number of parents of vertex i) data matrix
            I (numpy.ndarray): (n, 1) weighted adjacency vector for the i-th node
        Returns:
            numpy.ndarray: (s, 1) data matrix.
        '''

        #N = np.random.uniform(low=-1.0, high=1.0, size=s)
        mu, sigma = 0, 1
        N = np.random.normal(mu, sigma, size=s)
        #loc, scale = 0., 1.
        #N = np.random.laplace(loc, scale, size=s)
        return X @ I + N

    X = np.zeros([s, n])
    G = nx.DiGraph(A)

    ''' Radomly set ill conditioned nodes'''
    nodes = np.arange(n)
    np.random.shuffle(nodes)


    for v in list(nx.topological_sort(G)):
        P = list(G.predecessors(v))
        X[:, v] = _SEM(X[:, P], A[P, v])

    return X
