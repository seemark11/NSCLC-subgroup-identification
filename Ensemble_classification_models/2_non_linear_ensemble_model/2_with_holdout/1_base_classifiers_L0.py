#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd  
import numpy as np
import time
from functools import reduce

from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
from sklearn.model_selection import GridSearchCV
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier

#%%
n_jobs = 8
# IP data path
ip_data = "Data/data_base_classifiers.csv"

# OP paths
report_op_path = "classifier_reports.csv"
req_res_path = "classifier_results.csv"

svm_bst_fit_rbf_model_path = "svm_rbf.pkl"
rbf_train_prob_path = "svm_rbf_train_pred_prob.csv"
rbf_valid_prob_path = "svm_rbf_valid_pred_prob.csv"
rbf_test_prob_path = "svm_rbf_test_pred_prob.csv"

svm_bst_fit_lin_model_path = "svm_lin.pkl"
lin_train_prob_path = "svm_lin_train_pred_prob.csv"
lin_valid_prob_path = "svm_lin_valid_pred_prob.csv"
lin_test_prob_path = "svm_lin_test_pred_prob.csv"

svm_bst_fit_poly_model_path = "svm_poly.pkl"
poly_train_prob_path = "svm_poly_train_pred_prob.csv"
poly_valid_prob_path = "svm_poly_valid_pred_prob.csv"
poly_test_prob_path = "svm_poly_test_pred_prob.csv"

rf_bst_fit_gini_model_path = "rf_gini.pkl"
gini_train_prob_path = "rf_gini_train_pred_prob.csv"
gini_valid_prob_path = "rf_gini_valid_pred_prob.csv"
gini_test_prob_path = "rf_gini_test_pred_prob.csv"

rf_bst_fit_entropy_model_path = "rf_entropy.pkl"
entropy_train_prob_path = "rf_entropy_train_pred_prob.csv"
entropy_valid_prob_path = "rf_entropy_valid_pred_prob.csv"
entropy_test_prob_path = "rf_entropy_test_pred_prob.csv"

ffnn_op_train_file = "ffnn_train_pred_prob.csv"
ffnn_op_valid_file = "ffnn_valid_pred_prob.csv"
ffnn_op_test_file = "ffnn_test_pred_prob.csv"

checkpoint_filepath = "Weights/pcg_{epoch:02d}_{val_accuracy:.2f}.hdf5"
t = time.time()
export_path_keras = "Weights/pcg_{}.h5".format(int(t))


#%%
ip_df = pd.read_csv(ip_data, index_col = 0)
ip_df.iloc[0:5, 0:5]
ip_df.head()

ip_df['Clusters'] = ip_df['Clusters'] - 1
#%%
# How many ones
print("\n Cluster zero samples " +str(ip_df[ip_df.Clusters == 0].shape))
print("\n Cluster one samples " +str( ip_df[ip_df.Clusters == 1].shape))
print("\n Cluster two samples " +str( ip_df[ip_df.Clusters == 2].shape))
print("\n Cluster three samples " +str( ip_df[ip_df.Clusters == 3].shape))
print("\n Cluster four samples " +str( ip_df[ip_df.Clusters == 4].shape))
print("\n Cluster samples \n" +str( ip_df['Clusters'].value_counts()))

# Separate features and labels
X = ip_df.drop('Clusters', axis=1)  
y = ip_df['Clusters']

print("\n Input shape: " +str(X.shape))
print("\n Output shape: " +str(y.shape))

# Train-test split
x_train, X_test, y_train, Y_test = train_test_split(X, y, test_size = 0.1, random_state = 121, shuffle = True, stratify = y)
# Further split train into train and validation
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.4, random_state = 121, shuffle = True, stratify = y_train)

#%%
# Parameters for SVM training
Cs = [0.001, 0.01, 0.1, 1, 5, 10, 50, 100]
gammas = [0.001, 0.01, 0.1, 1, 5, 10, 50]
degrees = [1, 2, 3, 4, 5, 6, 7, 8]

#%%
# Sigma for rbf kernel
def svc_rbf_param_selection(X, y, nfolds):
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(estimator = SVC(kernel='rbf', probability = True, random_state = 121), param_grid = param_grid, n_jobs = n_jobs, cv = nfolds, verbose = 1)
    grid_search.fit(X, y)
    print("================================\n")
    print("SVM RBF\n")
    print("Best parameters: " + str(grid_search.best_params_))
    print("================================\n")
    return (grid_search.best_estimator_)

#%%
def svc_lin_param_selection(X, y, nfolds):
    param_grid = {'C': Cs}
    grid_search = GridSearchCV(estimator = SVC(kernel='linear', probability = True, random_state = 121), param_grid = param_grid, n_jobs = n_jobs, cv = nfolds, verbose = 1)
    grid_search.fit(X, y)
    print("================================\n")
    print("SVM Linear\n")
    print("Best parameters: " + str(grid_search.best_params_))
    print("================================\n")
    return (grid_search.best_estimator_)

#%%
def svc_poly_param_selection(X, y, nfolds):
    param_grid = {'degree': degrees}
    grid_search = GridSearchCV(estimator = SVC(kernel='poly', probability = True, random_state = 121), param_grid = param_grid, n_jobs = n_jobs, cv = nfolds, verbose = 1)
    grid_search.fit(X, y)
    print("================================\n")
    print("SVM Polynomial\n")
    print("Best parameters: " + str(grid_search.best_params_))
    print("================================\n")
    return (grid_search.best_estimator_)

#%%
def eval_metrics(fit_model, x_data, y_true, key_str):
    
    # Obtain prediction probabilities
    pred_prob = round(pd.DataFrame(fit_model.predict_proba(x_data)), 6)
    
    # Obtain evalution metrices
    y_obtained = fit_model.predict(x_data)
    print("Confusion Matrix \n")
    print(confusion_matrix(y_true, y_obtained))
    print("======================\n")
    eval_report = round((pd.DataFrame(classification_report(y_true, y_obtained, output_dict = True))*100).transpose(), 3)
    eval_report.columns = eval_report.columns + '_' + key_str

    return (pred_prob, eval_report)
    
#%%
svm_bst_fit_rbf_model = svc_rbf_param_selection((X_train), Y_train, 5)
joblib.dump(svm_bst_fit_rbf_model, svm_bst_fit_rbf_model_path)

# Obtain prediction probabilities and evalution metrices
print("======== Train ========\n")
pred_rbf_train_prob, svm_rbf_train_report = eval_metrics(svm_bst_fit_rbf_model, (X_train), Y_train, "train_rbf_svm")
pred_rbf_train_prob.set_index(X_train.index, inplace=True)
pred_rbf_train_prob.to_csv(rbf_train_prob_path, index = True, header = True)

print("======== Validation ========\n")
pred_rbf_valid_prob, svm_rbf_valid_report = eval_metrics(svm_bst_fit_rbf_model, (X_valid), Y_valid, "valid_rbf_svm")
pred_rbf_valid_prob.set_index(X_valid.index, inplace=True)
pred_rbf_valid_prob.to_csv(rbf_valid_prob_path, index = True, header = True)

print("======== Test ========\n")
pred_rbf_test_prob, svm_rbf_test_report = eval_metrics(svm_bst_fit_rbf_model, (X_test), Y_test, "test_rbf_svm")
pred_rbf_test_prob.set_index(X_test.index, inplace=True)
pred_rbf_test_prob.to_csv(rbf_test_prob_path, index = True, header = True)


#%%
svm_bst_fit_lin_model = svc_lin_param_selection((X_train), Y_train, 5)
joblib.dump(svm_bst_fit_lin_model, svm_bst_fit_lin_model_path)

# Obtain prediction probabilities and evalution metrices
print("======== Train ========\n")
pred_lin_train_prob, svm_lin_train_report = eval_metrics(svm_bst_fit_lin_model, (X_train), Y_train, "train_lin_svm")
pred_lin_train_prob.set_index(X_train.index, inplace=True)
pred_lin_train_prob.to_csv(lin_train_prob_path, index = True, header = True)

print("======== Validation ========\n")
pred_lin_valid_prob, svm_lin_valid_report = eval_metrics(svm_bst_fit_lin_model, (X_valid), Y_valid, "valid_lin_svm")
pred_lin_valid_prob.set_index(X_valid.index, inplace=True)
pred_lin_valid_prob.to_csv(lin_valid_prob_path, index = True, header = True)

print("======== Test ========\n")
pred_lin_test_prob, svm_lin_test_report = eval_metrics(svm_bst_fit_lin_model, (X_test), Y_test, "test_lin_svm")
pred_lin_test_prob.set_index(X_test.index, inplace=True)
pred_lin_test_prob.to_csv(lin_test_prob_path, index = True, header = True)

#%%
svm_bst_fit_poly_model = svc_poly_param_selection((X_train), Y_train, 5)
joblib.dump(svm_bst_fit_poly_model, svm_bst_fit_poly_model_path)

# Obtain prediction probabilities and evalution metrices
print("======== Train ========\n")
pred_poly_train_prob, svm_poly_train_report = eval_metrics(svm_bst_fit_poly_model, (X_train), Y_train, "train_poly_svm")
pred_poly_train_prob.set_index(X_train.index, inplace=True)
pred_poly_train_prob.to_csv(poly_train_prob_path, index = True, header = True)

print("======== Validation ========\n")
pred_poly_valid_prob, svm_poly_valid_report = eval_metrics(svm_bst_fit_poly_model, (X_valid), Y_valid, "valid_poly_svm")
pred_poly_valid_prob.set_index(X_valid.index, inplace=True)
pred_poly_valid_prob.to_csv(poly_valid_prob_path, index = True, header = True)

print("======== Test ========\n")
pred_poly_test_prob, svm_poly_test_report = eval_metrics(svm_bst_fit_poly_model, (X_test), Y_test, "test_poly_svm")
pred_poly_test_prob.set_index(X_test.index, inplace=True)
pred_poly_test_prob.to_csv(poly_test_prob_path, index = True, header = True)

#%%
# Parameters for RF training
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 0.2, 0.3]

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [5, 10, 15]

# Minimum number of samples required at each leaf node
min_samples_leaf = [4, 10, 20]

#%%
def rf_gini_param_selection(X, y, nfolds):
    param_grid = {'n_estimators': n_estimators, 
                  'max_features': max_features, 
                   'max_depth': max_depth, 
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    
    grid_search = GridSearchCV(estimator = RandomForestClassifier(criterion = "gini", oob_score = True, random_state = 121), param_grid = param_grid, n_jobs = n_jobs, cv = nfolds, verbose = 0)
    grid_search.fit(X, y)
    
    print("================================\n")
    print("RF Gini\n")
    print("Best parameters: " + str(grid_search.best_params_))
    print("================================\n")
    return (grid_search.best_estimator_)

#%%
def rf_entropy_param_selection(X, y, nfolds):
   param_grid = {'n_estimators': n_estimators, 
                  'max_features': max_features, 
                   'max_depth': max_depth, 
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

   grid_search = GridSearchCV(estimator = RandomForestClassifier(criterion = "entropy", oob_score = True, random_state = 121), param_grid = param_grid, n_jobs = n_jobs, cv = nfolds, verbose = 0)
   grid_search.fit(X, y)
   
   print("================================\n")
   print("RF Entropy\n")
   print("Best parameters: " + str(grid_search.best_params_))
   print("================================\n")
   return (grid_search.best_estimator_)


#%%
rf_bst_fit_gini_model = rf_gini_param_selection((X_train), Y_train, 5)
joblib.dump(rf_bst_fit_gini_model, rf_bst_fit_gini_model_path)

# Obtain prediction probabilities and evalution metrices
print("======== Train ========\n")
pred_gini_train_prob, rf_gini_train_report = eval_metrics(rf_bst_fit_gini_model, (X_train), Y_train, "train_gini_rf")
pred_gini_train_prob.set_index(X_train.index, inplace=True)
pred_gini_train_prob.to_csv(gini_train_prob_path, index = True, header = True)

print("======== Validation ========\n")
pred_gini_valid_prob, rf_gini_valid_report = eval_metrics(rf_bst_fit_gini_model, (X_valid), Y_valid, "valid_gini_rf")
pred_gini_valid_prob.set_index(X_valid.index, inplace=True)
pred_gini_valid_prob.to_csv(gini_valid_prob_path, index = True, header = True)

print("======== Test ========\n")
pred_gini_test_prob, rf_gini_test_report = eval_metrics(rf_bst_fit_gini_model, (X_test), Y_test, "test_gini_rf")
pred_gini_test_prob.set_index(X_test.index, inplace=True)
pred_gini_test_prob.to_csv(gini_test_prob_path, index = True, header = True)


#%%
rf_bst_fit_entropy_model = rf_entropy_param_selection((X_train), Y_train, 5)
joblib.dump(rf_bst_fit_entropy_model, rf_bst_fit_entropy_model_path)

# Obtain prediction probabilities and evalution metrices
print("======== Train ========\n")
pred_entropy_train_prob, rf_entropy_train_report = eval_metrics(rf_bst_fit_entropy_model, (X_train), Y_train, "train_entropy_rf")
pred_entropy_train_prob.set_index(X_train.index, inplace=True)
pred_entropy_train_prob.to_csv(entropy_train_prob_path, index = True, header = True)

print("======== Validation ========\n")
pred_entropy_valid_prob, rf_entropy_valid_report = eval_metrics(rf_bst_fit_entropy_model, (X_valid), Y_valid, "valid_entropy_rf")
pred_entropy_valid_prob.set_index(X_valid.index, inplace=True)
pred_entropy_valid_prob.to_csv(entropy_valid_prob_path, index = True, header = True)

print("======== Test ========\n")
pred_entropy_test_prob, rf_entropy_test_report = eval_metrics(rf_bst_fit_entropy_model, (X_test), Y_test, "test_entropy_rf")
pred_entropy_test_prob.set_index(X_test.index, inplace=True)
pred_entropy_test_prob.to_csv(entropy_test_prob_path, index = True, header = True)

#%%
reports = [svm_rbf_train_report, svm_rbf_valid_report, svm_rbf_test_report, svm_lin_train_report, svm_lin_valid_report, svm_lin_test_report, svm_poly_train_report, svm_poly_valid_report, svm_poly_test_report, rf_gini_train_report, rf_gini_valid_report, rf_gini_test_report, rf_entropy_train_report, rf_entropy_valid_report, rf_entropy_test_report]


df_final = reduce(lambda left,right: pd.merge(left, right, left_index = True, right_index = True), reports)
df_final.to_csv(report_op_path, index = True, header = True)

#%%
# Function to save weights when there is reduction in validation loss
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only = True, verbose = 1, monitor = 'val_accuracy', mode = 'max', save_best_only = True)
# Early stopping function i.e stop training the model if the validation loss remains same for five subsequent epochs
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)
 
#%%
# Split valid into train and split required for FFNN
X_train_ffnn, X_valid_ffnn, Y_train_ffnn, Y_valid_ffnn = train_test_split(x_train, y_train, test_size = 0.1, random_state = 121, shuffle = True, stratify = y_train)

# Convert class labels to categorical data
Y_train_ffnn_categ = utils.to_categorical(Y_train_ffnn)
Y_valid_ffnn_categ = utils.to_categorical(Y_valid_ffnn)
Y_train_categ = utils.to_categorical(Y_train)
Y_valid_categ = utils.to_categorical(Y_valid)
Y_test_categ = utils.to_categorical(Y_test)

#%%
# Number of nodes in each layer and hyper-parameters
print("Declaring parameters \n")
inputs = X_train.shape[1]
hidden1 = 512
outputs = Y_train_categ.shape[1] 
learning_rate = 0.01
epochs = 200
batch_size = 16

param_grid = {'learning_rate': [0.05, 0.01, 0.005, 0.001]}

print("Parameters Declared \n")
print(f'Input Nodes: {inputs} , Hidden 1 nodes : {hidden1} , Output nodes : {outputs} \n')
print(f'Learning rate: {learning_rate} , Epochs : {epochs} , Batch size : {batch_size} \n')
print("Building and compiling model \n")

#%%

def ffnn_vanila(activation = 'relu', learning_rate = 0.01):    
 	# define model
    model = Sequential()
    model.add(Input(shape = inputs))
    model.add(Dense(units = hidden1, activation = activation, kernel_initializer = 'he_normal', bias_initializer = 'zeros', name = "hidden_layer_1"))             
    model.add(Dense(units = outputs, activation = 'softmax', kernel_initializer = 'glorot_normal', bias_initializer = 'zeros', name = "output_layer"))    
    #=============================================================== 
    opt_adam = Adam(learning_rate = learning_rate)
    model.compile(optimizer = opt_adam, loss = 'categorical_crossentropy', metrics = ["accuracy"])
    #=============================================================== 
    # Prints a description of the model  
    print(model.summary())
    return model

#%%
model = KerasClassifier(build_fn = ffnn_vanila)
grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = n_jobs, cv = 5)
grid_result = grid.fit((X_train), Y_train_categ)
print("Best fit parameters \n")
print(grid_result.best_params_)
print("\n ====================== \n")

#%%
# fit model
ffnn_model = ffnn_vanila(learning_rate = grid_result.best_params_['learning_rate'])
ffnn_history = ffnn_model.fit((X_train_ffnn), Y_train_ffnn_categ, batch_size = batch_size, epochs = epochs, shuffle = True, validation_data = ((X_valid_ffnn), Y_valid_ffnn_categ), callbacks=[early_stopping_callback, model_checkpoint_callback])

#%%
ffnn_model.save(export_path_keras)
[train_loss, train_acc] = ffnn_model.evaluate((X_train_ffnn), Y_train_ffnn_categ)

print("Train Original")
[train_loss, train_acc] = ffnn_model.evaluate((X_train), Y_train_categ)
train_acc = round((train_acc)*100, 3)

print("Validation")
[valid_loss, valid_acc] = ffnn_model.evaluate((X_valid), Y_valid_categ)
valid_acc = round((valid_acc)*100, 3)

print("Test")
[test_loss, test_acc] = ffnn_model.evaluate((X_test), Y_test_categ)
test_acc = round((test_acc)*100, 3)
#%%
ffnn_train_prob_df = round(pd.DataFrame(ffnn_model.predict((X_train)), index = X_train.index), 6)
print(ffnn_train_prob_df.shape)
print(ffnn_train_prob_df.iloc[0:5, 0:5])
ffnn_train_prob_df.to_csv(ffnn_op_train_file, index=True) 

#%%
ffnn_valid_prob_df = round(pd.DataFrame(ffnn_model.predict((X_valid)), index = X_valid.index), 6)
print(ffnn_valid_prob_df.shape)
print(ffnn_valid_prob_df.iloc[0:5, 0:5])
ffnn_valid_prob_df.to_csv(ffnn_op_valid_file, index=True) 

#%%
ffnn_test_prob_df = round(pd.DataFrame(ffnn_model.predict((X_test)), index = X_test.index), 6)
print(ffnn_test_prob_df.shape)
print(ffnn_test_prob_df.iloc[0:5, 0:5])
ffnn_test_prob_df.to_csv(ffnn_op_test_file, index=True) 

#%%
# Tabulate the results
req_res = pd.DataFrame([[rf_entropy_train_report.loc['accuracy',:][1], rf_entropy_valid_report.loc['accuracy',:][1], rf_entropy_test_report.loc['accuracy',:][1]],
 [rf_gini_train_report.loc['accuracy',:][1], rf_gini_valid_report.loc['accuracy',:][1], rf_gini_test_report.loc['accuracy',:][1]],
 [svm_lin_train_report.loc['accuracy',:][1], svm_lin_valid_report.loc['accuracy',:][1], svm_lin_test_report.loc['accuracy',:][1]],
 [svm_poly_train_report.loc['accuracy',:][1], svm_poly_valid_report.loc['accuracy',:][1], svm_poly_test_report.loc['accuracy',:][1]],
 [svm_rbf_train_report.loc['accuracy',:][1], svm_rbf_valid_report.loc['accuracy',:][1], svm_rbf_test_report.loc['accuracy',:][1]],
 [train_acc, valid_acc, test_acc]],
 columns = ["Train", "Valid", "Test"], index = ["rf_entropy", "rf_gini", "svm_lin", "svm_poly", "svm_rbf", "ffnn"])
req_res.dtypes
req_res = req_res.T
req_res.to_csv(req_res_path, index = True, header = True)
