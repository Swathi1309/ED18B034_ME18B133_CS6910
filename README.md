# Image classification using MLP
Execute the first five sections of the code: 
  Install and Initialization of packages
  Loading dataset and example images
  Parameters initializing, activation and loss functions
  Feedforward and Backpropagation
  Optimizers

## Question 4
We have executed five iterations of gridsearch to find the best model. We have specified the parameterdictionaries for each iteration. Run them one-by-one along with adjoining sweep codes to perform the sweeps in wandb interface. Analysing the charts and tables in wandb, we have identified the best model.

## Question 6
The following interesting observations have been made from the parallel plots, scaterred plots and accuracy charts:
  

## Question 7
For the confusion matrix, run the corresponding cells in the code.

## Question 8
Modify the loss function(Loss) and the derivative of loss function(dLoss) by adding 'l2' string in the end of the input arguments of these functions. Now run the commands in the cell given the same question to get log its values into wandb. Check the validation_accuracy of this model and our best model to get an idea of the difference in performance.

## Question 10
The following hyperparameters are observed to be critical for model selection:
  
  
Hence we would use combinations of these. The three combinations to run grid search are:
  
  
  
