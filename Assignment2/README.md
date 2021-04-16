# Assignment 2
## Part A
- We have defined two models with the same structure- one with batch normalization and the other one without it. Initialize the models using one of the following codes.
- With and without Batch Normalization respectively
```
CNN_w_BN(filter_matrix, kernel_matrix, activation_matrix, fdropout, dense_no, learning_rate)
CNN_wo_BN(filter_matrix, kernel_matrix, activation_matrix, fdropout, dense_no, learning_rate)
```
> Note: Run the Define Parameters cell and choose one filter matrix and kernel matrix  configuration before initializing the model.

-The results on test data, visualization of filters and guided backpropagation using the best model can be obtained by running the appropriately named cells in the code.

## Part B
- The Xception model pre-trained on ImageNet dataset has been fine tuned and used here. The data and model have been loaded using the following commands.
```
load_model(dropout, learning_rate, unfreeze)
load_data(dir_train, dir_test, batch)
```
> Note: Run the Define Parameters before initializing the model.
- Run the hyperparameter sweeps to obtain the best model.
