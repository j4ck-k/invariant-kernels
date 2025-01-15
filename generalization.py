import numpy as np
from sage.all import *

def elementary_symmetric_polynomials(x, m_degree, d_dimension):

    mat = np.zeros((m_degree+1, d_dimension+1))

    mat[0] = np.ones(d_dimension+1)
    
    for i in range(m_degree):
        for j in range(i, d_dimension):
            mat[i+1][j+1] = x[j]*mat[i][j] + mat[i+1][j]

    return mat[:, d_dimension]

def esp_basis(x, m_degree, d_dimension):

    parts = []
    for i in range(m_degree + 1):
        parts.extend(Partitions(i).list())

    esp_x = elementary_symmetric_polynomials(x, m_degree, d_dimension)
    basis_x = [1]

    for i in range(1, len(parts)):
        part = parts[i]
        basis_x.append(np.prod([esp_x[p] for p in part]))

    return np.array(basis_x)


test_mse = np.zeros((4, 10, 10))
test_pe = np.zeros((4, 10, 10))


for m in range(2, 6):
    print(f'Degree {m}')
    
    rank = np.sum([Partitions(i).cardinality() for i in range(m+1)])

    for i in range(10):
        train_data = np.load(f'Data/Poly-Generalization/noisy_train_d100_m{m}_poly{i}.npz')
        train_X = train_data['X']
        train_y = train_data['y']

        mat = np.zeros((train_X.shape[0], rank))

        for j in range(train_X.shape[0]):
            mat[j] = esp_basis(train_X[j], m, train_X.shape[1])

        alpha = np.linalg.solve(mat.T @ mat, mat.T @ train_y)

        for k in range(1, 11):
            d = k*100
            test_data = np.load(f'Data/Poly-Generalization/noisy_test_d{d}_m{m}_poly{i}.npz')
            test_X = test_data['X']
            test_y = test_data['y']

            mat = np.zeros((test_X.shape[0], rank))
            for j in range(test_X.shape[0]):
                mat[j] = esp_basis(test_X[j], m, test_X.shape[1])

            test_mse[m-2][i][k-1] = np.means((mat@alpha - test_y)**2)
            test_pe[m-2][i][k-1] = np.mean(np.abs((mat@alpha - test_y)/test_y))
