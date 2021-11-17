
# Load required libraries

library(e1071) # SVM model
library(caret) # For confusion matrix
library(randomForest) # Random Forest
library(data.table) # Read and write files

#======================================================================

# Path to training data
train_ip_path <- "Data/Training_data_mean_sd_normalized.csv"

x_train <-  fread(train_op_path, header = TRUE, data.table = FALSE, stringsAsFactors = FALSE)
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

#======================================================================
#======================================================================

# Paths to required models
svm_model_rds <- "Results/svm_model.rds"
rf_model_rds <- "Results/randomForest_model.rds"

# Load the SVM models
model_svm <- readRDS(svm_model_rds)
print(model_svm)

# Load Random forest model
model_rf <- readRDS(rf_model_rds)
print(model_rf)

#======================================================================
#======================================================================

# Paths to store prediction probabilities
pred_prob_train <- "Results/Pred_prob_train.csv"
pred_prob_CoC_train <- "Results/CoC_prediction_train.csv"
acc_CoC_train <- "Results/Accuracy_CoC_train.csv"
acc_CoC_classwise_train <- "Results/Accuracy_classwise_CoC_train.csv"

#======================================================================
#======================================================================

# compute decision values and probabilities for SVM
pred_prob_svm_train <- predict(model_svm, x_train, type = "prob")
colnames(pred_prob_svm_train) <- paste("svm", colnames(pred_prob_svm_train), sep = "_")

# compute decision values and probabilities for Random forest
pred_prob_rf_train <- predict(model_rf, x_train, type = "prob")
colnames(pred_prob_rf_train) <- paste("rf", colnames(pred_prob_rf_train), sep = "_")

#======================================================================
# FFNN
#======================================================================
# Load results of FFNN
ffnn_train <- "Results/FFNN_train_prob.csv"

pred_prob_ffnn_train <- fread(ffnn_train, header = TRUE, data.table = FALSE, stringsAsFactors = FALSE)
# Make sample names as rownames and drop that column
rownames(pred_prob_ffnn_train) <- pred_prob_ffnn_train[,1]
pred_prob_ffnn_train <- pred_prob_ffnn_train[,-1]
colnames(pred_prob_ffnn_train) <- paste0("ffnn_C", seq(1:5))

#======================================================================
# Combine prediction probabilities from all the base classifiers
req_ip_CC <- cbind(pred_prob_svm_train,  pred_prob_rf_train, pred_prob_ffnn_train)
fwrite(as.data.frame(req_ip_CC), pred_prob_train, row.names = TRUE, col.names = TRUE)

#======================================================================
#======================================================================
# Generate the required weights combination
req_comb <- c()
alpha <- seq(0, 1, 0.05)
for (i in 1:length(alpha)) {
  alpha_1 <- alpha[i]
  beta <- seq(0, 1 - alpha_1, 0.1)
  for (j in 1:length(beta)) {
    beta_1 <- beta[j]
    gamma_1 = 1 - alpha_1 - beta_1
    req_comb <- rbind(req_comb, data.frame(alpha_1, beta_1, gamma_1))
  }
}

#======================================================================
#======================================================================

class_no <- 5 # Number of classes in the classification task
model_no <- 3 # Number of base classifiers

req_res_temp <- c()
req_acc <- data.frame(matrix(nrow = nrow(req_comb), ncol = 10))
req_acc_classwise <- data.frame(matrix(nrow = (nrow(req_comb)*class_no), ncol = 14))

# For each combination of alpha, beta and gamma
for (i in 1:nrow(req_comb)) {
  
  # create temp df for each o/p observation X no of classes
  temp_df <- data.frame(matrix(nrow = nrow(req_ip_CC), ncol = class_no))

  # Obtain prediction probability for each sample by weighing the base classifiers
  for (j in 1:class_no) {
    temp_df [,j] <- req_comb[i, 1] * req_ip_CC[, j] +
      req_comb[i, 2] * req_ip_CC[, (j + class_no)] +
      req_comb[i, 3] * req_ip_CC[, (j + 2*class_no)]
  }

  # Get the final class probability
  temp_df$class_prob = apply(temp_df, 1, max)
  temp_df$class = apply(temp_df [,1:class_no], 1, which.max)
  temp_df$class <- as.factor(paste0("C", temp_df$class))
  req_res_temp[[i]] <- temp_df

  # Get the confusion matrix
  cM_train <- confusionMatrix(data = temp_df$class, reference = y_train)
  cM_table_train <- cM_train[["table"]]
  cM_acc_train <- cM_train[["overall"]]
  cM_acc_by_class_train <- cM_train[["byClass"]]

  # Combine the weights and accuracies
  req_acc[i,] <- data.frame(req_comb[i,], t(cM_acc_train))
  # Save prediction probabilty for each class
  req_acc_classwise[((i+((i-1)*(class_no-1))):(class_no*i)),] <- data.frame(req_comb[i,], (cM_acc_by_class_train))
  rm(temp_df, cM_train, cM_table_train)

}

req_res_df <- Reduce(cbind, req_res_temp)
colnames(req_acc) <- c("alpha", "beta", "gamma", names(cM_acc_train))
colnames(req_acc_classwise) <- c("alpha", "beta", "gamma", colnames(cM_acc_by_class_train))

# Write the results
fwrite(as.data.frame(req_res_df), pred_prob_CoC_train, row.names = FALSE, col.names = TRUE)
fwrite(as.data.frame(req_acc), acc_CoC_train, row.names = FALSE, col.names = TRUE)
fwrite(as.data.frame(req_acc_classwise), acc_CoC_classwise_train, row.names = FALSE, col.names = TRUE)

#======================================================================
