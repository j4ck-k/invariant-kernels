# Invariant Kernels

This code accompanies the paper **Invariant Kernels: Rank Stabilization and Generalization Across Dimensions**. In order to use this code, SageMath must be installed.

The file `invariant_kernels.py` defines permutation and set-permutation invariant Taylor features kernels for regression problems and kernels to be used for binary classification of sets. The `Data` directory contains all data sets that were used in the experiments.

The file `experiments.py` defines each of the experiments contained in the paper. To replicate each of the experiments, run the relevant function:

- To replicate the two classification experiments, run `run_classification()`
- To replicate the generalization experiment, run `run_generalization()`
- To replicate the dimension estimation experiment, run `run_estimation()`

Each of the experiment types is defined as an object. To run the experiments with different parameters, initialize the relevant experiment object with the desired parameters. For example, to run the set classification experiment using kernels of degrees 2 and 3 only, run the command `ClassificationExperiment(degs=[3, 4])`.