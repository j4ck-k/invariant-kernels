from invariant_kernels import *
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sage.all import *
from scipy.linalg import block_diag

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

    tr= 0

    ev = np.linalg.eigvals(g)
    
    for k in range(m+1):
        comps = WeightedIntegerVectors(k, np.ones(ev.shape[0]))
        for comp in comps:
            tr += np.prod(np.power(ev, comp))

    return tr


class ClassificationExperiment:

    def __init__(self, 
                 var_degree=True,
                 degs = [1, 2, 3, 4, 5],
                 dims = [2, 3, 4, 5],
                 n_samples = [200, 400, 600, 800, 1000],
                 lams = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                 folds =4):
        
        self.n_samples = list(n_samples)
        self.lams = list(lams)
        self.folds = folds
        self.var_degree = var_degree
        self.degs = list(degs)
        self.dims = list(dims)

        if var_degree:
            data = np.load('Data/Classification/k2-Classification.npz')
            self.train_X = data['train_X']
            self.train_y = data['train_y']
            self.test_X = data['test_X']
            self.test_y = data['test_y']

            

    def cross_validation(self, folds = None):

        if folds is not None:
            self.folds = folds

        kfold = KFold(n_splits=self.folds)

        if self.var_degree:

            avg_val_accuracies = np.zeros((len(self.degs), len(self.n_samples)))
            best_lams = np.zeros((len(self.degs), len(self.n_samples)))
            inv_avg_val_accuracies = np.zeros((len(self.degs), len(self.n_samples)))
            inv_best_lams = np.zeros((len(self.degs), len(self.n_samples)))

            for i in range(len(self.degs)):

                inv_kernel = SetInvariantClassification(50, 2, self.degs[i])
                kernel = SetClassification(50, 2, self.degs[i])

                for j in range(len(self.n_samples)):
                    X = self.train_X[:self.n_samples[j]]
                    y = self.train_y[:self.n_samples[j]]

                    val_accuracies = np.zeros((self.folds, len(self.lams)))
                    inv_val_accuracies = np.zeros((self.folds, len(self.lams)))
                    
                    fold = 0

                    for train_index, test_index in kfold.split(X):

                        train_data = X[train_index]
                        train_values = y[train_index]
                        val_data = X[test_index]
                        val_values = y[test_index]

                        kernel_matrix = kernel.gram_matrix(train_data)
                        prediction_matrix = kernel.matrix(val_data, train_data)

                        inv_kernel_matrix = inv_kernel.gram_matrix(train_data)
                        inv_prediction_matrix = inv_kernel.matrix(val_data, train_data)

                        for l in range(len(self.lams)):
                            alpha = np.linalg.solve(kernel_matrix + self.lams[l]*np.eye(kernel_matrix.shape[0]), train_values)
                            val_accuracies[fold][l] = 1 - np.count_nonzero(val_values - np.sign(prediction_matrix@alpha))/val_data.shape[0]

                            alpha = np.linalg.solve(inv_kernel_matrix + self.lams[l]*np.eye(inv_kernel_matrix.shape[0]), train_values)
                            inv_val_accuracies[fold][l] = 1 - np.count_nonzero(val_values - np.sign(inv_prediction_matrix@alpha))/val_data.shape[0]

                        fold += 1

                    best_lams[i][j] = self.lams[np.argmax(np.mean(val_accuracies, axis=0))]
                    avg_val_accuracies[i][j] = np.max(np.mean(val_accuracies, axis=0))

                    inv_best_lams[i][j] = self.lams[np.argmax(np.mean(inv_val_accuracies, axis=0))]
                    inv_avg_val_accuracies[i][j] = np.max(np.mean(val_accuracies, axis=0))

        else:
            avg_val_accuracies = np.zeros((len(self.dims), len(self.n_samples)))
            best_lams = np.zeros((len(self.dims), len(self.n_samples)))
            inv_avg_val_accuracies = np.zeros((len(self.dims), len(self.n_samples)))
            inv_best_lams = np.zeros((len(self.dims), len(self.n_samples)))

            for i in range(len(self.dims)):

                inv_kernel = SetInvariantClassification(50, self.dims[i], self.degs[0])
                kernel = SetClassification(50, self.dims[i], self.degs[0])

                data = np.load(f'Data/Classification/k{i}-Classification.npz')
                train_X = data['train_X']
                train_y = data['train_y']

                for j in range(len(self.n_samples)):
                    X = train_X[:self.n_samples[j]]
                    y = train_y[:self.n_samples[j]]

                    val_accuracies = np.zeros((self.folds, len(self.lams)))
                    inv_val_accuracies = np.zeros((self.folds, len(self.lams)))
                    
                    fold = 0

                    for train_index, test_index in kfold.split(X):

                        train_data = X[train_index]
                        train_values = y[train_index]
                        val_data = X[test_index]
                        val_values = y[test_index]

                        kernel_matrix = kernel.gram_matrix(train_data)
                        prediction_matrix = kernel.matrix(val_data, train_data)

                        inv_kernel_matrix = inv_kernel.gram_matrix(train_data)
                        inv_prediction_matrix = inv_kernel.matrix(val_data, train_data)

                        for l in range(len(self.lams)):
                            alpha = np.linalg.solve(kernel_matrix + self.lams[l]*np.eye(kernel_matrix.shape[0]), train_values)
                            val_accuracies[fold][l] = 1 - np.count_nonzero(val_values - np.sign(prediction_matrix@alpha))/val_data.shape[0]

                            alpha = np.linalg.solve(inv_kernel_matrix + self.lams[l]*np.eye(inv_kernel_matrix.shape[0]), train_values)
                            inv_val_accuracies[fold][l] = 1 - np.count_nonzero(val_values - np.sign(inv_prediction_matrix@alpha))/val_data.shape[0]

                        fold += 1

                    best_lams[i][j] = self.lams[np.argmax(np.mean(val_accuracies, axis=0))]
                    avg_val_accuracies[i][j] = np.max(np.mean(val_accuracies, axis=0))

                    inv_best_lams[i][j] = self.lams[np.argmax(np.mean(inv_val_accuracies, axis=0))]
                    inv_avg_val_accuracies[i][j] = np.max(np.mean(val_accuracies, axis=0))

        self.inv_best_lams = inv_best_lams
        self.non_inv_best_lams = best_lams
        self.inv_val_accuracies = inv_avg_val_accuracies
        self.non_inv_val_accuracies = avg_val_accuracies

    def test(self):

        if self.var_degree:

            invariant_test_acc = np.zeros(len(self.degs), len(self.n_samples))
            noninvariant_test_acc = np.zeros(len(self.degs), len(self.n_samples))

            for i in range(len(self.degs)):

                kernel = SetClassification(50, 2, self.degs[i])
                invkernel = SetInvariantClassification(50, 2, self.degs[i])

                for j in range(len(self.n_samples)):
                    X = self.train_X[:self.n_samples[j]]
                    y = self.train_y[:self.n_samples[j]]

                    kernel.train(X, y, self.non_inv_best_lams[i][j])
                    invkernel.train(X, y, self.inv_best_lams[i][j])

                    invariant_test_acc[i][j] = invkernel.accuracy(self.test_X, self.test_y)
                    noninvariant_test_acc[i][j] = kernel.accuracy(self.test_X, self.test_y)

            else:
                invariant_test_acc = np.zeros(len(self.dims), len(self.n_samples))
                noninvariant_test_acc = np.zeros(len(self.dims), len(self.n_samples))

                for i in range(len(self.dims)):

                    data = np.load(f'Data/Classification/k{i}-Classification.npz')
                    train_X = data['train_X']
                    train_y = data['train_y']
                    test_X = data['test_X']
                    test_y = data['test_y']

                    kernel = SetClassification(50, self.dims[i], self.degs[0])
                    invkernel = SetInvariantClassification(50, self.dims[i], self.degs[0])

                    for j in range(len(self.n_samples)):
                        X = train_X[:self.n_samples[j]]
                        y = train_y[:self.n_samples[j]]

                        kernel.train(X, y, self.non_inv_best_lams[i][j])
                        invkernel.train(X, y, self.inv_best_lams[i][j])

                        invariant_test_acc[i][j] = invkernel.accuracy(test_X, test_y)
                        noninvariant_test_acc[i][j] = kernel.accuracy(test_X, test_y)


        self.invariant_test_acc = invariant_test_acc
        self.non_invariant_test_acc = noninvariant_test_acc

    def plot(self, save=True, save_file = 'Figures/classification.pdf'):

        if self.var_degree:
            for i in range(len(self.degs)):

                plt.plot(self.n_samples, self.invariant_test_acc[i], label=f'Degree {self.degs[i]}, Invariant', linewidth=2)
                plt.plot(self.n_samples, self.non_invariant_test_acc[i], label=f'Degree {self.degs[i]}, Non-Invariant', linewidth=2)

            plt.legend()
            plt.xlabel('Training Set Size')
            plt.xticks(self.n_samples)
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.ylabel('Accuracy')

            plt.legend(bbox_to_anchor=(1, 1))
            plt.savefig(save_file, bbox_inches='tight')

        else:
            for i in range(len(self.dims)):

                plt.plot(self.n_samples, self.invariant_test_acc[i], label=f'$k={self.dims[i]}$, Invariant', linewidth=2)
                plt.plot(self.n_samples, self.non_invariant_test_acc[i], label=f'$k= {self.dims[i]}$, Non-Invariant', linewidth=2)

            plt.legend()
            plt.xlabel('Training Set Size')
            plt.xticks(self.n_samples)
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.ylabel('Accuracy')

            plt.legend(bbox_to_anchor=(1, 1))

            if save:
                plt.savefig(save_file, bbox_inches='tight')

        fig = plt.gcf()
        return fig

class GeneralizationExperiment:

    def __init__(self, 
                 dims = range(100, 1001, 100), 
                 degs = [2, 3, 4, 5]):
        
        self.dims = dims
        self.degs = degs

    def test(self):

        test_mse = np.zeros((len(self.degs), len(self.dims), 10))
        test_pe = np.zeros((len(self.degs), len(self.dims), 10))


        for m in range(len(self.degs)):

            rank = np.sum([Partitions(i).cardinality() for i in range(m+1)])
            for i in range(10):
                train_data = np.load(f'Data/Poly-Generalization/train/noisy_train_d100_m{self.degs[m]}_poly{i}.npz')
                train_X = train_data['X']
                train_y = train_data['y']

                mat = np.zeros((train_X.shape[0], rank))

                for j in range(train_X.shape[0]):
                    mat[j] = esp_basis(train_X[j], m, train_X.shape[1])

                alpha = np.linalg.solve(mat.T @ mat, mat.T @ train_y)

                for k in len(self.dims):
                    d = self.dims[k]
                    test_data = np.load(f'Data/Poly-Generalization/d{d}/noisy_test_d{d}_m{m}_poly{i}.npz')
                    test_X = test_data['X']
                    test_y = test_data['y']

                    mat = np.zeros((test_X.shape[0], rank))
                    for j in range(test_X.shape[0]):
                        mat[j] = esp_basis(test_X[j], m, test_X.shape[1])

                    test_mse[m][k][i] = np.means((mat@alpha - test_y)**2)
                    test_pe[m][k][i] = np.mean(np.abs((mat@alpha - test_y)/test_y))

        self.test_mse = test_mse
        self.test_pe = test_pe

    def plot(self, mse=True, save=True, save_file='Figures/generalization-mse.pdf'):

        if mse:
            for i in range(len(self.degs)):
                mean_data = np.mean(self.test_mse[i], axis=1)
                plt.plot(self.dims, mean_data, label=f'Degree {self.degs[i]}')

            plt.ylabel('Mean Square Error')

        else:
            for i in range(len(self.degs)):
                mean_data = np.mean(self.test_pe[i], axis=1)
                plt.plot(self.dims, mean_data, label=f'Degree {self.degs[i]}')

            plt.ylabel('Mean Percent Error')

        plt.xlabel('Test Set Dimension')
        plt.legend(bbox_to_anchor=(1,1))

        if save:
            plt.savefig(save_file, bbox_inches='tight')

        fig=plt.gcf()
        return fig

class DimensionEstimation:

    def __init__(self, dimension, degree, samples=1000, repeats=10):
            
            self.dimension = dimension
            self.degree = degree
            self.n = samples
            self.r = repeats

    def estimate_graphs(self):

        G = SymmetricGroup(self.dimension)

        T = np.eye(self.dimension**2)
        plus = 0
        for i in range(self.dimension**2):
            T[i][i] = 0
            T[i][(plus + i//self.dimension)%self.dimension**2] = 1
            plus += self.dimension

        S = 0.5*(np.eye(self.dimension**2) + T)

        chars = np.zeros((self.r, self.n))
        estimates = np.zeros((self.r, self.n))
        for i in range(self.r):
            for j in range(self.n):
                m = G.random_element().matrix()
                chars[i][j]=(character(np.kron(m, m)@S, self.degree))
                estimates[i][j] = np.mean(chars[i][:j+1])

        self.graph_estimates = estimates
        return estimates
    
    def estimate_permutations(self):

        G = SymmetricGroup(self.dimension)
        chars = np.zeros((self.r, self.n))
        estimates = np.zeros((self.r, self.n))

        for i in range(self.r):
            for j in range(self.n):
                chars[i][j]=(character(G.random_element().matrix(), self.degree))
                estimates[i][j] = np.mean(chars[i][:j+1])

        self.permutation_estimates = estimates
        return estimates
    
    def estimate_orthogonal(self):

        chars = np.zeros((self.r, self.n))
        estimates = np.zeros((self.r, self.n))

        for i in range(self.r):
            for j in range(self.n):
                chars[i][j]=(character(random_orthogonal(self.dimension), self.degree))
                estimates[i][j] = np.mean(chars[i][:j+1])

        self.orthogonal_estimates = estimates
        return estimates
    
    def plot(self, group, save=True, save_file='Figures/dimension-estimate.pdf'):

        if group == 'permutations':
            data = self.permutation_estimates
        elif group == 'graphs':
            data = self.graph_estimates
        else:
            data = self.orthogonal_estimates

        mean_data = np.mean(data, axis=0)
        plt.plot(range(1, self.n+1), mean_data)
        plt.ylabel('Dimension Estimate')
        plt.xlabel('Number of Samples')

        if save:
            plt.savefig(save_file)

        fig = plt.gcf()
        return fig

def run_classification():

    deg_exp = ClassificationExperiment()
    deg_exp.cross_validation()
    deg_exp.test()
    deg_exp.plot(save_file='Figures/classification-var-deg.pdf')

    dim_exp = ClassificationExperiment(False, 2)
    dim_exp.cross_validation()
    dim_exp.test()
    dim_exp.plot(save_file='Figures/classification-var-dim.pdf')

def run_generalization():

    exp = GeneralizationExperiment()
    exp.test()
    exp.plot(save_file='Figures/generalization-mse.pdf')
    exp.plot(False, save_file='Figures/generalization-pe.pdf')

def run_estimation():

    exp = DimensionEstimation(5, 4)
    exp.estimate_orthogonal()
    exp.estimate_permutations()
    exp.plot('orthogonal', save_file='Figures/dimension-orthogonal.pdf')
    exp.plot('permutations', save_file='Figures/dimension-permutation.pdf')

    exp=DimensionEstimation(5, 2)
    exp.graph_estimates()
    exp.plot('graphs', save_file='Figures/dimension-graphs.pdf')








       




        
            
