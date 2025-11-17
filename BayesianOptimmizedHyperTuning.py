from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # why : StandardScaler? To standardize features by removing the mean and scaling to unit variance
import pandas as pd 
from xgboost import XGBClassifier # why : XGBClassifier? To use the XGBoost algorithm for classification tasks
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna as op

data = load_breast_cancer()
X, y= data.data, data.target

X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Displaying the data (before standardization)
print (f"traing data = {X_train}") 
print (f"Testing data = {X_test}")
print(" ")


#standerdized feauture
Scaler = StandardScaler() # why : StandardScaler? To standardize features by removing the mean and scaling to unit variance
X_train = Scaler.fit_transform(X_train) # why : fit_transform? To fit the scaler on the training data and then transform it
X_test = Scaler.transform(X_test)  # why : transform? To apply the same transformation to the test data

print(f"Training data shape :{X_train.shape}")
print(f"Testing data shape :{X_test.shape}")

# Displaying the data (after standardization)
print (f"traing data = {X_train}") 
print (f"Testing data = {X_test}")

#Train a baseline XGBoost model
baseline_model = XGBClassifier(eval_metric='logloss', random_state=42 ) # why : eval_metric='logloss'? To specify the evaluation metric for the model
baseline_model.fit(X_train, y_train) # why : fit? To train the model on the training data

#Evaluate the baseline model
baseline_preds = baseline_model.predict(X_test) # why : predict? To make predictions on the test data
baseline_accuracy = accuracy_score(y_test, baseline_preds) # why : accuracy_score? To calculate the accuracy of the model
print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")

# define the objective function for optuna
def objective(trial): # why : objective function? To define the function that Optuna will optimize
    #suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500), # why : suggest_int? To suggest an integer value for the hyperparameter |why : n_estimators? Number of trees in the model
        'max_depth': trial.suggest_int('max_depth', 3, 100), # why : max_depth? Maximum depth of a tree | To control overfitting
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), # why : learning_rate? Step size shrinkage used to prevent overfitting 
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), # why : subsample? Fraction of samples to be used for fitting the individual base learners
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), # why : colsample_bytree? Fraction of features to be used for fitting the individual base learners
        'gamma': trial.suggest_float('gamma', 0, 5), # why : gamma? Minimum loss reduction required to make a further partition on a leaf node of the tree
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10), # why : reg_alpha? L1 regularization term on weights
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10), # why : reg_lambda? L2 regularization term on weights
    }
    
    # train xgboost model with the suggested hyperparameters
    model = XGBClassifier(**params, eval_metric='logloss', random_state=42) # why : **params? To unpack the hyperparameters dictionary
    model.fit(X_train, y_train) # why : fit? To train the model on the training data
    
    #Evaluate the model on the validation set
    preds = model.predict(X_test) # why : predict? To make predictions on the test data
    accuracy = accuracy_score(y_test, preds) # why : accuracy_score? To calculate the accuracy of the model
    return accuracy

#create an optuna study
study = op.create_study(direction='maximize') # why : direction='maximize'? To maximize the objective function (accuracy in this case) | To create a study object for optimization
study.optimize(objective, n_trials=50) # why : n_trials=50? To specify the number of trials for the optimization process | To start the optimization process

#Get the best hyperparameters
print("Best Hyperparameters: ", study.best_params) # why : best_params? To get the best hyperparameters found during the optimization
print("Best Accuracy: ", study.best_value) # why : best_value? To get the best accuracy achieved during the optimization

