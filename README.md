# Image classification using MLP
Execute the first five sections of the code:

* Install and Initialization of packages (including logging in to wandb)
* Loading dataset and example images
* Parameters initializing, activation and loss functions
* Feedforward and Backpropagation
* Optimizers

For each of the questions, a code cell from the section **Commands for execution** can be run.

## Question 1
Load the dataset (with augmented images) using:
```python
X_train, Y_train, X_validation, Y_validation, X_test, Y_test = load_data()
```
Next, run the `log_images()` function to plot one random sample image from each class to wandb.

## Question 2
The feedforward network has been implemented in the `feedforward()` function. To view an example of the predcited outputs for an image, run the code snippet under Question 2.

## Question 4
Five iterations of gridsearch have been executed to find the best model. The parameter dictionaries for each iteration have been specified. Run them one-by-one along with adjoining sweep codes to perform the sweeps in wandb interface. Analysing the charts and tables in wandb, the best model has been identified.
  
## Question 7
For the confusion matrix of the best model obtained above, run the corresponding cells in the code.

## Question 8
Modify the loss function `Loss()` and the derivative of loss function `dLoss()` by adding 'l2' string as the last argument where these functions are called (in `loss_and_accuracy()` and `feedforward()` functions). Now run the commands in the cell under **Question 8** to log its values into wandb. Check the `validation_accuracy` of this model and our best model to get an idea of the difference in performance.

## Question 10
The three hyperparameter combinations to run for the MNIST dataset are given in the parameters dictionary. Execute the cells under **Question 10** to load the MNIST data set, and to train the model and log the accuracies and losses in Wandb.
