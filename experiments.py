from invariant_kernels import *
import numpy as np
from sklearn.model_selection import KFold

# Classifying Point Clouds from Skew Distributions

data = np.load('Data/XY-Skew-Train.npz')
train_X = data['x']
train_y = data['y']

degs = list(range(1, 6))
n_samples = [200, 400, 600, 800, 1000]
lams = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
folds  = 4

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

        np.save('Data/XY-Non-Invariant-Classification-val-acc.npy', avg_val_accuracies)
        np.save('Data/XY-Non-Invariant-Classification-lambda.npy', best_lams)
        np.save('Data/XY-Invariant-Classification-val-acc.npy', inv_avg_val_accuracies)
        np.save('Data/XY-Invariant-Classification-lambda.npy', inv_best_lams)

invariant_test_acc = np.zeros((5, 5))
noninvariant_test_acc = np.zeros((5, 5))

data = np.load('Data/XY-Skew-Test.npz')
test_X = data['x']
test_y = data['y']

for i in range(len(degs)):

    kernel = SetClassification(50, 2, degs[i])
    invkernel = SetInvariantClassification(50, 2, degs[i])

    for j in range(len(n_samples)):
        print(f'{n_samples[j]} Samples')
        X = train_X[:n_samples[j]]
        y = train_y[:n_samples[j]]

        if invariant_test_acc[i][j] == 0:

            kernel.train(X, y, 1e-5)
            invkernel.train(X, y, 1e-5)

            invariant_test_acc[i][j] = invkernel.accuracy(test_X, test_y)
            noninvariant_test_acc[i][j] = kernel.accuracy(test_X, test_y)

            np.save('Data/XY-Invariant-Classification-test-acc.npy', invariant_test_acc)
            np.save('Data/XY-Non-Invariant-Classification-test-acc.npy', noninvariant_test_acc)


# Standard Normal vs Random Distribution Classification

data = np.load('Data/SND-Random-Classification-Train.npz')
train_X = data['x']
train_y = data['y']

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

        np.save('Data/SND-Random-Non-Invariant-val-acc.npy', avg_val_accuracies)
        np.save('Data/SND-Random-Non-Invariant-lambda.npy', best_lams)
        np.save('Data/SND-Random-Classification-val-acc.npy', inv_avg_val_accuracies)
        np.save('Data/SND-Random-Classification-lambda.npy', inv_best_lams)

invariant_test_acc = np.zeros((5, 5))
noninvariant_test_acc = np.zeros((5, 5))

data = np.load('Data/SND-Random-Classification-Test.npz')
test_X = data['x']
test_y = data['y']

for i in range(len(degs)):

    print(f'Degree {degs[i]}')
    kernel = SetClassification(50, 2, degs[i])
    invkernel = SetInvariantClassification(50, 2, degs[i])

    for j in range(len(n_samples)):
        print(f'{n_samples[j]} Samples')
        X = train_X[:n_samples[j]]
        y = train_y[:n_samples[j]]

        if invariant_test_acc[i][j] == 0:

            kernel.train(X, y, 1e-5)
            invkernel.train(X, y, 1e-5)

            invariant_test_acc[i][j] = invkernel.accuracy(test_X, test_y)
            noninvariant_test_acc[i][j] = kernel.accuracy(test_X, test_y)

            print('Invariant Accuracy \n', invariant_test_acc)
            print('Non-Invariant Accuracy \n', noninvariant_test_acc)

            np.save('Data/SND-Random-Classification-test-acc.npy', invariant_test_acc)
            np.save('Data/SND-Random-Non-Invariant-test-acc.npy', noninvariant_test_acc)