import numpy as np
from sage.all import *
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

def householder(v):

    d = v.shape[0]
    x = v[0]
    u = np.atleast_2d((v + np.sign(x)*np.eye(d)[:, 0])/np.linalg.norm(v + np.sign(x)*np.eye(d)[:, 0])).T

    return -np.sign(x) * (np.eye(d) - 2 * u @ u.T)

def embed(M, d):
    return block_diag(np.eye(d - M.shape[0]), M)

def random_orthogonal(d):

    O = np.eye(d)

    for i in range(1, d+1):
        v = np.random.randn(i)
        H = embed(householder(v), d)

        O = H @ O

    return O

def character(g, m):
    
    tr = 1

    ev = np.linalg.eigvals(g)
    
    for k in range(1, m+1):
        comps = WeightedIntegerVectors(k, np.ones(ev.shape[0]))
        for comp in comps:
            tr += np.prod(np.power(ev, comp))

    return tr

chars = [character(random_orthogonal(3), 4) for _ in range(1000)]
ortho_estimates = [np.mean(chars[:i+1]) for i in range(1000)]


G = SymmetricGroup(5)
chars = [character(G.random_element().matrix(), 4) for _ in range(1000)]
perm_estimates = [np.mean(chars[:i+1]) for i in range(1000)]

chars = []
graph_estimates = []
for _ in range(1000):
    g = G.random_element().matrix()
    chars.append(character(np.kron(g, g), 4))
    graph_estimates.append(np.mean(chars))

plt.plot(range(1000), ortho_estimates, color='#648FFF', label='Estimate')
plt.axhline(3.0, color='#DD2680', label='True Value')

plt.xlabel('Number of Samples')
plt.ylabel('Dimension')
plt.legend()
plt.title('Orthogonal Transformations')
plt.savefig('Figures/orthogonal-estimate.pdf')

plt.clf()

plt.plot(range(1000), perm_estimates, color='#648FFF', label='Estimate')
plt.axhline(12, color='#DD2680', label='True Value')

plt.xlabel('Number of Samples')
plt.ylabel('Dimension')
plt.legend()
plt.title('Coordinate Permutations')
plt.savefig('Figures/permutation-estimate.pdf')

plt.clf()

plt.plot(range(1000), graph_estimates, color='#648FFF', label='Estimate')

plt.xlabel('Number of Samples')
plt.ylabel('Dimension')
plt.legend()
plt.title('Graph Permutations')
plt.savefig('Figures/graph-estimate.pdf')

