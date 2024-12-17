import numpy as np

def snd_vs_random_data(n_samples, d_points, k):

    X = np.zeros((n_samples, d_points, k))
    y = 2*np.random.binomial(1, 0.5, n_samples) - 1

    for i in range(n_samples):
        pc = np.random.randn(d_points, k)

        if y[i] == 1:
            X[i] = pc
        
        else:
            M = np.random.randn(k, k)
            X[i] = pc @ (M@M.T)

    
    return X, y

def xy_skew_data(n_samples, d_points):

    X = np.zeros((n_samples, d_points, 2))
    y = 2*np.random.binomial(1, 0.5, n_samples) - 1

    for i in range(n_samples):
        pc = np.random.randn(d_points, 2)

        if y[i] == 1:
            X[i] = pc @np.array([[1 ,0], [0, 2]])
        
        else:
            X[i] = pc @ np.array([[2 ,0], [0, 1]])

    return X, y