from invariant_kernels import SetInvariantClassification, SetClassification
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Testing Kernels of Varying Degree on 2D Data

data = np.load('Data/Classification/k2-Classification.npz')
train_X = data['train_X']
train_y = data['train_y']
test_X = data['test_X']
test_y = data['test_y']

degs = list(range(1, 6))
n_samples = [200, 400, 600, 800, 1000]
lams = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
folds  = 4

# Finding optimal regularizer

kfold = KFold(n_splits=folds)

avg_val_accuracies = np.zeros((5, 5))
best_lams = np.zeros((5, 5))
inv_avg_val_accuracies = np.zeros((5, 5))
inv_best_lams = np.zeros((5, 5))

for i in range(len(degs)):

    inv_kernel = SetInvariantClassification(50, 2, degs[i])
    kernel = SetClassification(50, 2, degs[i])

    for j in range(len(n_samples)):
        X = train_X[:n_samples[j]]
        y = train_y[:n_samples[j]]

        val_accuracies = np.zeros((folds, len(lams)))
        inv_val_accuracies = np.zeros((folds, len(lams)))
        
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

            for l in range(len(lams)):
                alpha = np.linalg.solve(kernel_matrix + lams[l]*np.eye(kernel_matrix.shape[0]), train_values)
                val_accuracies[fold][l] = 1 - np.count_nonzero(val_values - np.sign(prediction_matrix@alpha))/val_data.shape[0]

                alpha = np.linalg.solve(inv_kernel_matrix + lams[l]*np.eye(inv_kernel_matrix.shape[0]), train_values)
                inv_val_accuracies[fold][l] = 1 - np.count_nonzero(val_values - np.sign(inv_prediction_matrix@alpha))/val_data.shape[0]

            fold += 1

        best_lams[i][j] = lams[np.argmax(np.mean(val_accuracies, axis=0))]
        avg_val_accuracies[i][j] = np.max(np.mean(val_accuracies, axis=0))

        inv_best_lams[i][j] = lams[np.argmax(np.mean(inv_val_accuracies, axis=0))]
        inv_avg_val_accuracies[i][j] = np.max(np.mean(val_accuracies, axis=0))

# Test Accuracy

invariant_test_acc = np.zeros((5, 5))
noninvariant_test_acc = np.zeros((5, 5))

for i in range(len(degs)):

    kernel = SetClassification(50, 2, degs[i])
    invkernel = SetInvariantClassification(50, 2, degs[i])

    for j in range(len(n_samples)):
        print(f'{n_samples[j]} Samples')
        X = train_X[:n_samples[j]]
        y = train_y[:n_samples[j]]

        kernel.train(X, y, best_lams[i][j])
        invkernel.train(X, y, inv_best_lams[i][j])

        invariant_test_acc[i][j] = invkernel.accuracy(test_X, test_y)
        noninvariant_test_acc[i][j] = kernel.accuracy(test_X, test_y)

# Creating Plot

plt.plot(n_samples, invariant_test_acc[0], '-o', color='#648FFF', label='Degree 1, Invariant', linewidth=2)
plt.plot(n_samples, noninvariant_test_acc[0], '-.o', color='#648FFF', label='Degree 1, Non-Invariant', linewidth=2)

plt.plot(n_samples, invariant_test_acc[1], '-D', color='#775EF0', label='Degree 2, Invariant', linewidth=2)
plt.plot(n_samples, noninvariant_test_acc[1], '-.D', color='#775EF0', label='Degree 2, Non-Invariant', linewidth=2)

plt.plot(n_samples, invariant_test_acc[2], '-^', color='#DD2680', label='Degree 3, Invariant', linewidth=2)
plt.plot(n_samples, noninvariant_test_acc[2], '-.^', color='#DD2680', label='Degree 3, Non-Invariant', linewidth=2)

plt.plot(n_samples, invariant_test_acc[3], '-s', color='#FE6100', label='Degree 4, Invariant', linewidth=2)
plt.plot(n_samples, noninvariant_test_acc[3], '-.s', color='#FE6100', label='Degree 4, Non-Invariant', linewidth=2)

plt.plot(n_samples, invariant_test_acc[4], '-p', color='#FFB001', label='Degree 5, Invariant')
plt.plot(n_samples, noninvariant_test_acc[4], '-.p', color='#FFB001', label='Degree 5, Non-Invariant')

plt.legend()
plt.xlabel('Training Set Size')
plt.xticks([200, 400, 600, 800, 1000])
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.ylabel('Accuracy')

plt.legend(bbox_to_anchor=(1, 1))
plt.savefig('Figures/classification.pdf', bbox_inches='tight')

