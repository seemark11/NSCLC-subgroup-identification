
# Train the base models with 5-fold CV repeated 10 times to obtain optimal hyper-parameters

# Load required libraries
library(e1071) # SVM model
library(caret) # For confusion matrix
library(randomForest) # Random Forest
library(data.table) # Read and write files

#===============================================================================
# Models
#===============================================================================
# Path to training data
train_ip_path <- "Data/Training_data_mean_sd_normalized.csv"

x_train <-  fread(train_ip_path, header = TRUE, data.table = FALSE, stringsAsFactors = FALSE)
x_train[1:5, 1:5]
# Make sample names as rownames and drop that column
rownames(x_train) <- x_train[,1]
x_train <- x_train[,-1]
x_train[1:5, 1:5]
# Get the labels to train the model and convert it to factors
y_train <- as.factor(x_train$Clusters)
# Drop labels from training data
x_train <- subset(x_train, select = -Clusters)
x_train[1:5, 1:5]

#===============================================================================
#===============================================================================
# Save trained model as rds object
svm_model_rds <- "Results/svm_model.rds"
rf_model_rds <- "Results/randomForest_model.rds"

#===================================================================

#===============================================================================
# SVM
#===============================================================================
# 5 folds repeat 10 times and search method as grid
Sys.time()
set.seed(123)
control <- trainControl(method = 'repeatedcv',
                        classProbs =  TRUE,
                        number = 5, # 5-fold
                        repeats = 10, # repeated 10 times
                        verboseIter = TRUE)

system.time(model_svm <- train(x_train, y_train,
                               method = 'svmRadial', # SVM with radial kernel
                               metric = 'Accuracy', # choose the model with best accuracy
                               trControl = control))
Sys.time()
# Check the hyper parameters and respective accuracies
print(model_svm)
# Save the trained model
saveRDS(model_svm, svm_model_rds)

#===============================================================================

#===============================================================================
# Random Forest
#===============================================================================
# 5 folds repeat 10 times
Sys.time()
set.seed(123)
control <- trainControl(method = 'repeatedcv',
                        classProbs =  TRUE,
                        number = 5, # 5-fold
                        repeats = 10, # repeated 10 times
                        verboseIter = TRUE)

system.time(model_rf <- train(x_train, y_train,
                              method = 'rf', # random forest
                              metric = 'Accuracy',  # choose the model with best accuracy
                              trControl = control))
Sys.time()
# Check the hyper parameters and respective accuracies
print(model_rf)
# Save the trained model
saveRDS(model_rf, rf_model_rds)

#===============================================================================
