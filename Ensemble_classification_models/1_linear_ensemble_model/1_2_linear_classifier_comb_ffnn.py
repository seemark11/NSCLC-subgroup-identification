# -*- coding: utf-8 -*-

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.layers import PReLU

#%%
import matplotlib.pyplot as plt
import pandas as pd
import copy 
import time
from sklearn.model_selection import train_test_split

#%%
# Input and output file paths
ip_testing_file = "Data/test_data_min_max.csv"
ip_training_file = "Data/training_data_min_max.csv"

file_name_loss = "ffnn_loss.pdf"
file_name_acc = "ffnn_acc.pdf"

op_test_file = "prob_ffnn_test.csv"
op_train_file = "prob_ffnn_train.csv"

checkpoint_filepath = "Weights/ffnn_{epoch:02d}_{val_accuracy:.2f}.hdf5"
t = time.time()
export_path_keras = "Weights/ffnn_{}.h5".format(int(t))

#%%
print("Reading data \n")
training_data = pd.read_csv(ip_training_file, index_col=0) 
testing_data = pd.read_csv(ip_testing_file, index_col=0)

# Convert class labels to categorical data
train_labels = utils.to_categorical(training_data["Clusters"].str.slice(1).astype(int)-1)
# Drop class label from training data
X_train = training_data.drop('Clusters', axis=1)

# Convert class labels to categorical data
test_labels = utils.to_categorical(testing_data["Clusters"].str.slice(1).astype(int)-1)
# Drop class label from testing data
X_test = testing_data.drop('Clusters', axis=1)

print(train_labels.shape)
print(X_train.shape)

print(test_labels.shape)
print(X_test.shape)

X_train_orig = X_train
train_labels_orig = train_labels

test_labels[1:5]
X_test.iloc[1:5, 1:5]

# Split data to obtain validation dataset
X_train, X_valid, train_labels, valid_labels = train_test_split(X_train, train_labels, test_size=0.10, shuffle=True, random_state = 1)

print(train_labels.shape)
print(X_train.shape)

print(valid_labels.shape)
print(X_valid.shape)
#%%
# Function to save weights when there is reduction in validation loss
model_checkpoint_callback = ModelCheckpoint(filepath = checkpoint_filepath, 
                                            save_weights_only = True, 
                                            verbose = 1, 
                                            monitor = 'val_accuracy', 
                                            mode = 'max', 
                                            save_best_only = True)

# Early stopping function i.e stop training the model if the validation loss remains same for five subsequent epochs
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 5)

# Callback function required to save loss on test data in every epoch
class TestCallback(Callback):
    def __init__(self, test_data_x, test_data_y):
        self.x = test_data_x
        self.y = test_data_y
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        loss, acc = self.model.evaluate(self.x, self.y, verbose=0)
        self.losses.append(copy.deepcopy(loss))
        self.accuracy.append(copy.deepcopy(acc))
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
    def return_results(self):
        return self.losses, self.accuracy

# Create callback object
callbacks = TestCallback(X_test, test_labels)

#%%
# Number of nodes in each layer and hyper-parameters
print("Declaring parameters \n")
inputs, hidden1, hidden2 = X_train.shape[1], 128, 64
hidden3 = 32
outputs = train_labels.shape[1] 
learning_rate = 0.01
epochs = 200
batch_size = 20 # Change based on menory of GPU

print("Parameters Declared \n")
print(f'Input Nodes: {inputs} , Hidden 1 nodes : {hidden1} , Hidden 2 nodes: {hidden2} \n') 
print(f'Hidden 3 nodes : {hidden3} , Output nodes : {outputs} \n')
print(f'Learning rate: {learning_rate} , Epochs : {epochs} , Batch size : {batch_size} \n')
print("Building and compiling model \n")

# Define the model architecture 
def create_model():

    model = Sequential()

    model.add(Dense(units=hidden1, input_shape=(inputs,), kernel_initializer='he_normal', bias_initializer='zeros', name = 'input_layer',
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)))
    model.add(PReLU())
    model.add(Dropout(0.1))

    #=============================================================== 
    model.add(Dense(units=hidden2, kernel_initializer='he_normal', bias_initializer='zeros', name="hidden_layer_2",
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)))
    model.add(PReLU())
    model.add(Dropout(0.1))

    #=============================================================== 
    model.add(Dense(units=hidden3, kernel_initializer='he_normal', bias_initializer='zeros', name="hidden_layer_3",
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)))
    model.add(PReLU())

    #=============================================================== 
    model.add(Dense(units=outputs, activation='softmax',kernel_initializer='glorot_normal', bias_initializer='zeros', name="output_layer",
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)))
    
    #=============================================================== 
    opt_adam = Adam(learning_rate=learning_rate)  
    model.compile(optimizer=opt_adam, loss='categorical_crossentropy', metrics=["accuracy"])

    #=============================================================== 

    #model.summary prints a description of the model
    model.summary()
    return model

#%%
# Create and train the model
train_model = create_model()

history_model = train_model.fit(X_train, train_labels, batch_size=batch_size, epochs=epochs, 
                          shuffle=True, validation_data=(X_valid, valid_labels), 
                          callbacks=[early_stopping_callback, callbacks, model_checkpoint_callback])
# Save the trained model
train_model.save(export_path_keras)

#%%
# Get accuracy and loss for test data on each epoch - reqired for plotting
test_loss_obt, test_acc_obt = callbacks.return_results()

# Get accuracy and loss values
print("Train Original")
[train_loss, train_acc] = train_model.evaluate(X_train_orig, train_labels_orig)
print("Test")
[test_loss, test_acc] = train_model.evaluate(X_test, test_labels)

# Obtain the prediction probabilities
train_prob_df = pd.DataFrame(train_model.predict(X_train_orig), index = X_train_orig.index)
print(train_prob_df.shape)
print(train_prob_df.iloc[0:5, 0:5])
train_prob_df.to_csv(op_train_file, index=True)

test_prob_df = pd.DataFrame(train_model.predict(X_test), index = X_test.index)
print(test_prob_df.shape)
print(test_prob_df.iloc[0:5, 0:5])
test_prob_df.to_csv(op_test_file, index=True)

#%%
print("Plotting figures\n")

# Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history_model.history['loss'],'r',linewidth=3.0)
plt.plot(history_model.history['val_loss'],'b',linewidth=3.0)
plt.plot(test_loss_obt,'g',linewidth=3.0)
plt.legend(['Training Loss', 'Validation Loss', 'Testing Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig(file_name_loss, bbox_inches='tight')
plt.close()

# Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history_model.history['accuracy'],'r',linewidth=3.0)
plt.plot(history_model.history['val_accuracy'],'b',linewidth=3.0)
plt.plot(test_acc_obt,'g',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy', 'Testing Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig(file_name_acc, bbox_inches='tight')
plt.close()