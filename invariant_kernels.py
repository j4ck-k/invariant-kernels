from sage.all import binomial, WeightedIntegerVectors, factorial, Partitions
import numpy as np

from scipy.sparse import block_diag
from numpy.linalg import inv
from sklearn.model_selection import KFold
from tqdm import tqdm

def L(r):

    r = np.roll(r, -1)
    r[-1] = 0
    return r.astype(int)

def binomials(r, s):
    return np.prod([binomial(int(r[i]), int(s[i])) for i in range(r.shape[0])])

def red(r, s):
    return (r - s + L(s)).astype(int)

def gale_ryser(p, q):

    if np.sum(p) != np.sum(q):
        return False
    else:

        passed = True

        q_prime = np.array([np.count_nonzero(q >= i) for i in range(q.shape[0])])
        for j in range(q.shape[0]):
            if np.sum(q_prime[:j]) < np.sum(p[:j]):
                passed = False

        return passed
    
def comps_lessthan(k, s):

    n = s.shape[0]
    comps = WeightedIntegerVectors(k, np.ones(n))
    comps_less = []
    for comp in comps:
        if np.all(np.less_equal(comp, s)):
            comps_less.append(np.array(comp).astype(int))

    return comps_less

def un_bar(q_bar):

    q = np.concatenate(tuple([i+1 * np.ones(q_bar[i]) for i in range(q_bar.shape[0])]))
    return np.sort(q)[::-1]

def N(p , q):

    # print(f'finding number of matrices for p = {p}, q = {q}')

    if np.sum(p) != np.sum(q):
        return 0
    
    elif np.sum(p) == 0:
        return 1
    
    else:
        n = np.sum(p)

        # Check Gale-Ryser Criterion
        if gale_ryser(p, q):
            # Find vector of counts
            q_bar = np.array([np.count_nonzero(q == i) for i in range(1, n+1)]).astype(int)

            # Find compositions of p_1 less than or equal to q_bar
            comps = comps_lessthan(p[0], q_bar)

            num = 0
            for comp in comps:
                num += binomials(q_bar, comp) * N(L(p), un_bar(red(q_bar, comp)))
            return num
        
        else:
            return 0 


class InvariantTF:

    def __init__(self, d, n):
        self.degree = n
        self.dimension = d
        self.alpha = None
        self.train_data = None

        self.parts = []
        for i in range(self.degree + 1):
            self.parts.extend(Partitions(i).list())

        if n < 11:
            self.C = np.load(f'C-Matrices/C_{n}.npy')

        else:
            B = inv(self.change_basis())
            self.C = B @ self.C_star() @ B

    def C_star(self):
        rank = len(self.parts)

        C = np.zeros((rank, rank))
        C[0][0] = 1

        for i in range(1, rank):
            part  = np.array(self.parts[i])
            size = int(np.sum(np.array(part)))

            coeff = np.prod([factorial(np.count_nonzero(part == i)) for i in range(size)]) * factorial(size)/np.prod([factorial(int(part[i])) for i in range(len(part))])
            
            C[i][i] = coeff/factorial(size)

        #print('found C*')
        return C
    
    def change_basis(self):

        blocks = [np.ones((1, 1))]
        #print('Changing basis')

        for i in range(1, len(self.parts)):
            P = self.parts[i]
            mat = np.zeros((len(P), len(P)))

            for i in range(len(P)):
                for j in range(len(P)):
                    v = N(np.array(list(P[i])), np.array(list(P[j])))
                    mat[i][j] = v
                    mat[j][i] = v

            blocks.append(mat)

        return block_diag(tuple(blocks)).toarray()
    

    def elementary_symmetric_polynomials(self, x):

        mat = np.zeros((self.degree+1, self.dimension+1))

        mat[0] = np.ones(self.dimension+1)
        
        for i in range(self.degree):
            #print(f'row {i}')
            for j in range(i, self.dimension):
                mat[i+1][j+1] = x[j]*mat[i][j] + mat[i+1][j]

        return mat[:, self.dimension]
    
    def evaluate(self, x, y):

        esp_x = self.elementary_symmetric_polynomials(x)
        esp_y = self.elementary_symmetric_polynomials(y)

        #print('ESP Computed')

        basis_x = [1]
        basis_y = [[1]]

        for i in range(1, len(self.parts)):
            part = self.parts[i]
            basis_x.append(np.prod([esp_x[p] for p in part]))
            basis_y.append([np.prod([esp_y[p] for p in part])])

        return (np.array(basis_y).T @ self.C @ np.array(basis_x))[0]
    
    def matrix(self, X, Y):

        rank = len(self.parts)

        esp_X = np.zeros((X.shape[0], self.degree + 1))
        esp_Y = np.zeros((Y.shape[0], self.degree + 1))

        for i in range(X.shape[0]):
            esp_X[i] = self.elementary_symmetric_polynomials(X[i])

        for i in range(Y.shape[0]):
            esp_Y[i] = self.elementary_symmetric_polynomials(Y[i])

        basis_X = np.zeros((rank, X.shape[0]))
        basis_Y = np.zeros((rank, Y.shape[0]))

        basis_X[0] = np.ones(X.shape[0])
        basis_Y[0] = np.ones(Y.shape[0])

        for i in range(1, rank):
            part = self.parts[i]

            basis_X[i] = np.array([np.prod([esp_X[j][p] for p in part]) for j in range(X.shape[0])])
            basis_Y[i] = np.array([np.prod([esp_Y[j][p] for p in part]) for j in range(Y.shape[0])])

        return basis_Y.T @ self.C @ basis_X
    
    def gram_matrix(self, X):
        N = X.shape[0]
        mat = np.zeros((N, N))

        for i in range(N):
            for j in range(i, N):
                v = self.evaluate(X[i], X[j])
                mat[i][j] = v
                mat[j][i] = v

        return mat

    def predict(self, X):

        P = self.matrix(self.train_data, X)

        return P @ self.alpha

    def rrse(self, X, Y):

        return np.sqrt(np.sum((Y-self.predict(X))**2)/np.sum((Y - self.mean_value)**2))
    
    def relative_error(self, X, Y):

        return np.mean(((Y - self.predict(X))/(Y))**2)

    def train(self, X, y, lam):

        N = X.shape[0]
        M = self.gram_matrix(X)

        alpha = np.linalg.solve(M + lam * np.eye(N), y)

        self.alpha = alpha
        self.train_data = X
        self.train_values = y
        self.mean_value = np.mean(y)
        self.training_error = self.relative_error(X, y)
    
    def mse(self, X, y):
        
        return np.mean((y - self.predict(X))**2)
    
class InvariantInnerProduct:

    def __init__(self, d, n):
        self.dimension = d
        self.degree = int(n) 

        self.parts = Partitions(n).list()

        if n < 11:
            self.C = np.load(f'C-Matrices/IP-C-Matrices/C_{n}.npy')

        else:
            B = inv(self.change_basis())
            self.C = B @ self.C_star() @ B

    def C_star(self):
        rank = len(self.parts)

        C = np.zeros((rank, rank))

        for i in range(rank):
            part  = np.array(self.parts[i])
            size = int(np.sum(np.array(part)))

            coeff = np.prod([factorial(np.count_nonzero(part == i)) for i in range(size)])/np.prod([factorial(int(part[i])) for i in range(len(part))])
            
            C[i][i] = coeff

        #print('found C*')
        return C
    
    def change_basis(self):

        mat = np.zeros((len(self.parts), len(self.parts)))

        for i in range(len(self.parts)):
            for j in range(len(self.parts)):
                v = N(np.array(list(self.parts[i])), np.array(list(self.parts[j])))
                mat[i][j] = v
                mat[i][j] = v

        return mat
    

    def elementary_symmetric_polynomials(self, x):

        mat = np.zeros((self.degree+1, self.dimension+1))

        mat[0] = np.ones(self.dimension+1)
        
        for i in range(self.degree):
            #print(f'row {i}')
            for j in range(i, self.dimension):
                mat[i+1][j+1] = x[j]*mat[i][j] + mat[i+1][j]

        return mat[:, self.dimension]
    
    def evaluate(self, x, y):

        esp_x = self.elementary_symmetric_polynomials(x)
        esp_y = self.elementary_symmetric_polynomials(y)

        #print('ESP Computed')

        basis_x = []
        basis_y = []

        for i in range(len(self.parts)):
            part = self.parts[i]
            basis_x.append(np.prod([esp_x[p] for p in part]))
            basis_y.append([np.prod([esp_y[p] for p in part])])

        return (np.array(basis_y).T @ self.C @ np.array(basis_x))[0]
    
class SetInvariantClassification:

    def __init__(self, d, k, n):
        self.d = d
        self.k = k
        self.degree = n
        
        self.comps = []
        self.ips = []
        for i in range(self.degree + 1):
            self.comps.extend(WeightedIntegerVectors(i, np.ones(k)).list())
            self.ips.append(InvariantInnerProduct(d, i))

    def evaluate(self, x, y):
        
        val = 0
        for i in range(len(self.comps)):
            comp = self.comps[i]

            inner_products = [self.ips[int(comp[j])].evaluate(x[:, j], y[:, j]) for j in range(self.k)]
            val += np.prod(inner_products)/np.prod([factorial(int(c)) for c in comp])
            
        return val

    def matrix(self, X, Y):

        mat = np.zeros((X.shape[0], Y.shape[0]))  
        for i in tqdm(range(X.shape[0])):
            #print(f'Row {i}')
            for j in range(Y.shape[0]):
            #    print(f'column {j}')
                mat[i][j] = self.evaluate(X[i], Y[j])

        return mat
    
    def gram_matrix(self, X):
        mat = np.zeros((X.shape[0], X.shape[0]))

        for i in tqdm(range(X.shape[0])):
            #print(f'Row {i}')
            for j in range(i, X.shape[0]):
                v = self.evaluate(X[i], X[j])
                mat[i][j] = v
                mat[j][i] = v

        return mat

    def predict(self, X):

        P = self.matrix(X, self.train_data)

        return np.sign(P @ self.alpha)

    def train(self, X, y, lam):

        N = X.shape[0]
        M = self.gram_matrix(X)

        alpha = np.linalg.solve(M + lam * np.eye(N), y)

        self.alpha = alpha
        self.train_data = X
        self.train_values = y
        self.training_accuracy = 1-np.count_nonzero(y - np.sign(M@alpha))/X.shape[0]
    
    def accuracy(self, X, y):    
        return 1-np.count_nonzero(y - self.predict(X))/X.shape[0]

class SetClassification:

    def __init__(self, d, k, n):
        self.d = d
        self.k = k
        self.degree = n

    def evaluate(self, x, y):
        
        x = x.flatten()
        y = y.flatten()
            
        return np.sum([(x@y)**(i+1)/factorial(i+1) for i in range(self.degree)])

    def matrix(self, X, Y):

        mat = np.zeros((X.shape[0], Y.shape[0]))  
        for i in tqdm(range(X.shape[0])):
            #print(f'Row {i}')
            for j in range(Y.shape[0]):
            #    print(f'column {j}')
                mat[i][j] = self.evaluate(X[i], Y[j])

        return mat
    
    def gram_matrix(self, X):
        mat = np.zeros((X.shape[0], X.shape[0]))

        for i in tqdm(range(X.shape[0])):
            #print(f'Row {i}')
            for j in range(i, X.shape[0]):
                v = self.evaluate(X[i], X[j])
                mat[i][j] = v
                mat[j][i] = v

        return mat

    def predict(self, X):

        P = self.matrix(X, self.train_data)

        return np.sign(P @ self.alpha)

    def train(self, X, y, lam):

        N = X.shape[0]
        M = self.gram_matrix(X)

        alpha = np.linalg.solve(M + lam * np.eye(N), y)

        self.alpha = alpha
        self.train_data = X
        self.train_values = y
        self.training_accuracy = 1-np.count_nonzero(y - np.sign(M@alpha))/X.shape[0]
    
    def accuracy(self, X, y):    
        return 1-np.count_nonzero(y - self.predict(X))/X.shape[0]


        


        
