#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd  
import time
from functools import reduce

from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
from sklearn.model_selection import GridSearchCV
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

from keras.wrappers.scikit_learn import KerasClassifier

#%%
# IP files
ip_data = "Data/data_base_classifiers.csv"

ffnn_train_path = "ffnn_train_pred_prob.csv"
ffnn_test_path = "ffnn_test_pred_prob.csv"
ffnn_valid_path = "ffnn_valid_pred_prob.csv"

rf_train_path = "rf_gini_train_pred_prob.csv"
rf_test_path = "rf_gini_test_pred_prob.csv"
rf_valid_path = "rf_gini_valid_pred_prob.csv"

svm_train_path = "svm_rbf_train_pred_prob.csv"
svm_test_path = "svm_rbf_test_pred_prob.csv"
svm_valid_path = "svm_rbf_valid_pred_prob.csv"

# OP files
log_reg_bst_fit_model_path = "L1_log_reg.pkl"
log_reg_train_prob_path = "L1_log_reg_train_pred_prob.csv"
log_reg_valid_prob_path = "L1_log_reg_valid_pred_prob.csv"
log_reg_test_prob_path = "L1_log_reg_test_pred_prob.csv"

ffnn_op_train_file = "L1_ffnn_train_pred_prob.csv"
ffnn_op_valid_file = "L1_ffnn_valid_pred_prob.csv"
ffnn_op_test_file = "L1_ffnn_test_pred_prob.csv"
checkpoint_filepath = "Weights/L1_{epoch:02d}_{val_accuracy:.2f}.hdf5"
t = time.time()
export_path_keras = "Weights/L1_{}.h5".format(int(t))

combined_report_path = "L1_classifier_reports.csv"
req_res_path = "L1_classifier_results.csv"

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
y = ip_df['Clusters']

# print("\n Input shape: " +str(X.shape))
print("\n Output shape: " +str(y.shape))

#%%
# Read prediction probablitiies
svm_train_prob = pd.read_csv(svm_train_path, index_col = 0)
svm_train_prob.columns = svm_train_prob.columns + "_svm"
svm_train_prob.iloc[0:5, 0:5]

svm_test_prob = pd.read_csv(svm_test_path, index_col = 0)
svm_test_prob.columns = svm_test_prob.columns + "_svm"
svm_test_prob.iloc[0:5, 0:5]

svm_valid_prob = pd.read_csv(svm_valid_path, index_col = 0)
svm_valid_prob.columns = svm_valid_prob.columns + "_svm"
svm_valid_prob.iloc[0:5, 0:5]

#==================================================================
rf_train_prob = pd.read_csv(rf_train_path, index_col = 0)
rf_train_prob.columns = rf_train_prob.columns + "_rf"
rf_train_prob.iloc[0:5, 0:5]

rf_test_prob = pd.read_csv(rf_test_path, index_col = 0)
rf_test_prob.columns = rf_test_prob.columns + "_rf"
rf_test_prob.iloc[0:5, 0:5]

rf_valid_prob = pd.read_csv(rf_valid_path, index_col = 0)
rf_valid_prob.columns = rf_valid_prob.columns + "_rf"
rf_valid_prob.iloc[0:5, 0:5]

#==================================================================
ffnn_train_prob = pd.read_csv(ffnn_train_path, index_col = 0)
ffnn_train_prob.columns = ffnn_train_prob.columns + "_ffnn"
ffnn_train_prob.iloc[0:5, 0:5]

ffnn_test_prob = pd.read_csv(ffnn_test_path, index_col = 0)
ffnn_test_prob.columns = ffnn_test_prob.columns + "_ffnn"
ffnn_test_prob.iloc[0:5, 0:5]

ffnn_valid_prob = pd.read_csv(ffnn_valid_path, index_col = 0)
ffnn_valid_prob.columns = ffnn_valid_prob.columns + "_ffnn"
ffnn_valid_prob.iloc[0:5, 0:5]

# Merge prediction probabilities
train_dfs = [svm_train_prob, rf_train_prob, ffnn_train_prob]
test_dfs = [svm_test_prob, rf_test_prob, ffnn_test_prob]
valid_dfs = [svm_valid_prob, rf_valid_prob, ffnn_valid_prob]

train_probs = reduce(lambda left,right: pd.merge(left, right, left_index = True, right_index = True), train_dfs)
train_probs.iloc[0:5, 0:5]

test_probs = reduce(lambda left,right: pd.merge(left, right, left_index = True, right_index = True), test_dfs)
test_probs.iloc[0:5, 0:5]

valid_probs = reduce(lambda left,right: pd.merge(left, right, left_index = True, right_index = True), valid_dfs)
valid_probs.iloc[0:5, 0:5]
#%%
train_probs = pd.merge(y, train_probs, left_index = True, right_index = True)
train_probs.iloc[0:5, 0:5]
train_probs_x = train_probs.drop(['Clusters'], axis = 1)
train_probs_y = train_probs['Clusters']

test_probs = pd.merge(y, test_probs, left_index = True, right_index = True)
test_probs_x = test_probs.drop(['Clusters'], axis = 1)
test_probs_y = test_probs['Clusters']

valid_probs = pd.merge(y, valid_probs, left_index = True, right_index = True)
valid_probs_x = valid_probs.drop(['Clusters'], axis = 1)
valid_probs_y = valid_probs['Clusters']

#%%
# Build logistic regression
Cs = [0.001, 0.01, 0.1, 0, 1, 5, 10, 50, 100]
penalties = ["l1", "l2", "elasticnet"]

#%%
# Sigma for rbf kernel
def log_reg_param_selection(X, y, nfolds):
    param_grid = {'C': Cs, 'penalty' : penalties}
    grid_search = GridSearchCV(estimator = LogisticRegression(random_state = 121), param_grid = param_grid, cv = nfolds, verbose = 1)
    grid_search.fit(X, y)
    print("================================\n")
    print("logistic regression\n")
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
log_reg_bst_fit_model = log_reg_param_selection(valid_probs_x, valid_probs_y, 5)
joblib.dump(log_reg_bst_fit_model, log_reg_bst_fit_model_path)

# Obtain prediction probabilities and evalution metrices
print("======== Train ========\n")
pred_log_reg_train_prob, log_reg_train_report = eval_metrics(log_reg_bst_fit_model, train_probs_x, train_probs_y, "train_log_reg")
pred_log_reg_train_prob.set_index(train_probs_x.index, inplace=True)
pred_log_reg_train_prob.to_csv(log_reg_train_prob_path, index = True, header = True)

print("======== Validation ========\n")
pred_log_reg_valid_prob, log_reg_valid_report = eval_metrics(log_reg_bst_fit_model, valid_probs_x, valid_probs_y, "valid_log_reg")
pred_log_reg_valid_prob.set_index(valid_probs_x.index, inplace=True)
pred_log_reg_valid_prob.to_csv(log_reg_valid_prob_path, index = True, header = True)

print("======== Test ========\n")
pred_log_reg_test_prob, log_reg_test_report = eval_metrics(log_reg_bst_fit_model, test_probs_x, test_probs_y, "test_log_reg")
pred_log_reg_test_prob.set_index(test_probs_x.index, inplace=True)
pred_log_reg_test_prob.to_csv(log_reg_test_prob_path, index = True, header = True)

#%%
model_checkpoint_callback = ModelCheckpoint(filepath = checkpoint_filepath, save_weights_only = True, verbose = 1, monitor = 'val_accuracy', mode = 'max', save_best_only = True)
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)
 
#%%
# Split valid into train and split required for FFNN
X_ffnn_train, X_ffnn_valid, Y_ffnn_train, Y_ffnn_valid = train_test_split(valid_probs_x, valid_probs_y, test_size = 0.1, random_state = 121, shuffle = True, stratify = valid_probs_y)

# Build ffnn
Y_ffnn_train_categ = utils.to_categorical(Y_ffnn_train)
Y_ffnn_valid_categ = utils.to_categorical(Y_ffnn_valid)

Y_train_categ = utils.to_categorical(train_probs_y)
Y_valid_categ = utils.to_categorical(valid_probs_y)
Y_test_categ = utils.to_categorical(test_probs_y)

# parameters
print("Declaring parameters \n")
inputs = X_ffnn_train.shape[1]
hidden1 = 8
outputs = Y_ffnn_train_categ.shape[1] 
learning_rate = 0.01
epochs = 200
batch_size = 5

param_grid = {'learning_rate': [0.05, 0.01, 0.005, 0.001]}

print("Parameters Declared \n")
print(f'Input Nodes: {inputs} , Hidden 1 nodes : {hidden1} , Output nodes: {outputs} \n') 
print(f'Learning rate: {learning_rate} , Epochs : {epochs} , Batch size : {batch_size} \n')
print("Building and compiling model \n")

#%%

def ffnn_vanila(activation = 'relu', learning_rate = 0.01):    
 	# define model
    model = Sequential()
    model.add(InputLayer(input_shape = inputs))
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
def eval_metrics_ffnn(fit_model, x_data, y_true, key_str):
    
    # Obtain prediction probabilities
    pred_prob = round(pd.DataFrame(fit_model.predict(x_data)), 6)
    
    # Obtain evalution metrices
    y_obtained = fit_model.predict(x_data)
    y_obtained = y_obtained.argmax(axis=1)
    print("Confusion Matrix \n")
    print(confusion_matrix(y_true, y_obtained))
    print("======================\n")
    eval_report = round((pd.DataFrame(classification_report(y_true, y_obtained, output_dict = True))*100).transpose(), 3)
    eval_report.columns = eval_report.columns + '_' + key_str
    return (pred_prob, eval_report)

#%%
model = KerasClassifier(build_fn = ffnn_vanila)
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5)
grid_result = grid.fit(valid_probs_x, Y_valid_categ)
print("Best fit parameters \n")
print(grid_result.best_params_)
print("\n ====================== \n")

#%%
# fit model
ffnn_model = ffnn_vanila(learning_rate = grid_result.best_params_['learning_rate'])
ffnn_history = ffnn_model.fit(X_ffnn_train, Y_ffnn_train_categ, batch_size = batch_size, epochs = epochs, shuffle = True, validation_data = (X_ffnn_valid, Y_ffnn_valid_categ), callbacks = [early_stopping_callback, model_checkpoint_callback])

#%%
ffnn_model.save(export_path_keras)
print("========================================================================\n")

print("Train Original")
[train_loss, train_acc] = ffnn_model.evaluate(train_probs_x, Y_train_categ)
train_acc = round((train_acc)*100, 3)

print("Validation")
[valid_loss, valid_acc] = ffnn_model.evaluate(valid_probs_x, Y_valid_categ)
valid_acc = round((valid_acc)*100, 3)

print("Test")
[test_loss, test_acc] = ffnn_model.evaluate(test_probs_x, Y_test_categ)
test_acc = round((test_acc)*100, 3)

#%%
# Obtain prediction probabilities and evalution metrices
print("======== Train ========\n")
pred_ffnn_train_prob, ffnn_train_report = eval_metrics_ffnn(ffnn_model, train_probs_x, train_probs_y, "train_ffnn")
pred_ffnn_train_prob.set_index(train_probs_x.index, inplace=True)
print(pred_ffnn_train_prob.shape)
print(pred_ffnn_train_prob.iloc[0:5, 0:5])
pred_ffnn_train_prob.to_csv(ffnn_op_train_file, index = True, header = True)

# Obtain prediction probabilities and evalution metrices
print("======== valid ========\n")
pred_ffnn_valid_prob, ffnn_valid_report = eval_metrics_ffnn(ffnn_model, valid_probs_x, valid_probs_y, "valid_ffnn")
pred_ffnn_valid_prob.set_index(valid_probs_x.index, inplace=True)
print(pred_ffnn_valid_prob.shape)
print(pred_ffnn_valid_prob.iloc[0:5, 0:5])
pred_ffnn_valid_prob.to_csv(ffnn_op_valid_file, index = True, header = True)

# Obtain prediction probabilities and evalution metrices
print("======== test ========\n")
pred_ffnn_test_prob, ffnn_test_report = eval_metrics_ffnn(ffnn_model, test_probs_x, test_probs_y, "test_ffnn")
pred_ffnn_test_prob.set_index(test_probs_x.index, inplace=True)
print(pred_ffnn_test_prob.shape)
print(pred_ffnn_test_prob.iloc[0:5, 0:5])
pred_ffnn_test_prob.to_csv(ffnn_op_test_file, index = True, header = True)

#%%
req_res = pd.DataFrame([[log_reg_train_report.loc['accuracy',:][1], log_reg_valid_report.loc['accuracy',:][1], log_reg_test_report.loc['accuracy',:][1]], [train_acc, valid_acc, test_acc]], columns = ["Train", "valid", "Test"], index = ["log_reg", "ffnn"])
req_res.dtypes
req_res = req_res.T
req_res.to_csv(req_res_path, index = True, header = True)

#%%
reports = [log_reg_train_report, log_reg_valid_report, log_reg_test_report, ffnn_train_report, ffnn_valid_report, ffnn_test_report]

combined_report = reduce(lambda left,right: pd.merge(left, right, left_index = True, right_index = True), reports)
combined_report.iloc[:, 0:5]
combined_report.to_csv(combined_report_path, index = True, header = True)
