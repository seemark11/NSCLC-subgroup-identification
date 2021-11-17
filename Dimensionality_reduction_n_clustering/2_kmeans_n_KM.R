
# Clustering and Kaplan-Meier analysis

library(data.table)
library(ConsensusClusterPlus)
library(survival)
library(survminer)
library(ggplot2)

#=====================================================================================

pdf_op_path <- "kmeans_op_folder"
ip_file_path <- "AE_reduced_dataset_1.csv"
pac_op_path <- "Consensus_PAC_values.csv"

#=====================================================================================

op_10 <- "Class_Consensus_Clustering_K_10.csv"
op_9 <- "Class_Consensus_Clustering_K_9.csv"
op_8 <- "Class_Consensus_Clustering_K_8.csv"
op_7 <- "Class_Consensus_Clustering_K_7.csv"
op_6 <- "Class_Consensus_Clustering_K_6.csv"
op_5 <- "Class_Consensus_Clustering_K_5.csv"
op_4 <- "Class_Consensus_Clustering_K_4.csv"
op_3 <- "Class_Consensus_Clustering_K_3.csv"
op_2 <- "Class_Consensus_Clustering_K_2.csv"

#=====================================================================================
# Reading i/p data 
compressed_data <- fread(ip_file_path, header = TRUE, data.table = FALSE, stringsAsFactors = FALSE)
compressed_data[1:5, 1:5]

#=====================================================================================
# Consensus clustering
# columns=items/samples and rows are features
res.ConClust = ConsensusClusterPlus(data.matrix(t(compressed_data)), # data to cluster
                                    maxK = 10, # maximum nuber of clusters
                                    reps = 1000, # number of times to repeat clustering
                                    pItem = 0.8, # percent of samples to consider for clustering in each iteration
                                    pFeature = 1, # percent of features to consider for clustering in each iteration
                                    plot = "pdf", # save clustering figures as pdf
                                    title = pdf_op_path, # path to save figures
                                    distance = "euclidean", # use euclidean distance
                                    clusterAlg = "km", # use k-means clustering algorithm
                                    seed = 2111133)


#=====================================================================================
# Write clustering results
cons_10 <- as.data.frame(res.ConClust[[10]][["consensusClass"]])
colnames(cons_10) <- ("Clusters")

cons_9 <- as.data.frame(res.ConClust[[9]][["consensusClass"]])
colnames(cons_9) <- ("Clusters")

cons_8 <- as.data.frame(res.ConClust[[8]][["consensusClass"]])
colnames(cons_8) <- ("Clusters")

cons_7 <- as.data.frame(res.ConClust[[7]][["consensusClass"]])
colnames(cons_7) <- ("Clusters")

cons_6 <- as.data.frame(res.ConClust[[6]][["consensusClass"]])
colnames(cons_6) <- ("Clusters")

cons_5 <- as.data.frame(res.ConClust[[5]][["consensusClass"]])
colnames(cons_5) <- ("Clusters")

cons_4 <- as.data.frame(res.ConClust[[4]][["consensusClass"]])
colnames(cons_4) <- ("Clusters")

cons_3 <- as.data.frame(res.ConClust[[3]][["consensusClass"]])
colnames(cons_3) <- ("Clusters")

cons_2 <- as.data.frame(res.ConClust[[2]][["consensusClass"]])
colnames(cons_2) <- ("Clusters")

#=====================================================================================

fwrite(cons_10, op_10, sep=",", row.names = TRUE, col.names = TRUE)
fwrite(cons_9, op_9, sep=",", row.names = TRUE, col.names = TRUE)
fwrite(cons_8, op_8, sep=",", row.names = TRUE, col.names = TRUE)
fwrite(cons_7, op_7, sep=",", row.names = TRUE, col.names = TRUE)
fwrite(cons_6, op_6, sep=",", row.names = TRUE, col.names = TRUE)
fwrite(cons_5, op_5, sep=",", row.names = TRUE, col.names = TRUE)
fwrite(cons_4, op_4, sep=",", row.names = TRUE, col.names = TRUE)
fwrite(cons_3, op_3, sep=",", row.names = TRUE, col.names = TRUE)
fwrite(cons_2, op_2, sep=",", row.names = TRUE, col.names = TRUE)

#=====================================================================================
# PAC implementation 
maxK <- 10
Kvec = 2:maxK
x1 = 0.1; x2 = 0.9 # threshold defining the intermediate sub-interval
PAC = rep(NA,length(Kvec))
names(PAC) = paste("K=",Kvec,sep="") # from 2 to maxK
for(i in Kvec){
  M = res.ConClust[[i]]$consensusMatrix
  Fn = ecdf(M[lower.tri(M)])
  PAC[i-1] = Fn(x2) - Fn(x1)
} # end for i

# The optimal K
optK = Kvec[which.min(PAC)]
cat(sprintf("Optimal K = %d\n", optK))
fwrite(as.data.frame(PAC), pac_op_path, row.names = TRUE, col.names = TRUE)

#=====================================================================================
# KM analysis
#=====================================================================================
# IP files
ip_clinical_path <- "Lung_LUAD_LUSC_combined_data/clinical_DFS_OS_for_KM.csv"
ip_cluster_path <- "Class_Consensus_Clustering_K_5.csv"

# Read clinical data
clinical_data <- fread(ip_clinical_path, header = TRUE, data.table = FALSE, stringsAsFactors = FALSE)
clinical_data[1:5, ]

# Read cluster labels
cons_res <- fread(ip_cluster_path, header = TRUE, data.table = FALSE, stringsAsFactors = FALSE)
cons_res[1:5, ]
colnames(cons_res)[1] <- "Samples"

# Combine clinical data and cluster results
Clinical_cluster <- merge(cons_res, clinical_data, by.x = "Samples", by.y = "Samples")

#=====================================================================================
# For overall survival
# Creating Survival Object
Surv_Obj <- Surv(Clinical_cluster$OS_MONTHS,
                 Clinical_cluster$OS_STATUS)

surv_diff <- survdiff(formula = Surv_Obj ~ Clinical_cluster$Clusters)
p_val <- (1 - pchisq(surv_diff$chisq, length(surv_diff$n) - 1))

fit <- do.call(survfit, list(formula = Surv_Obj ~ Clinical_cluster$Clusters, data = Clinical_cluster))

title = "Overall survival"
survp_os <- ggsurvplot(fit, legend = c(0.8,0.8), pval = TRUE, title = title,
                       surv.median.line="hv", 
                       surv.scale = "percent", ylab = "Surviving",  
                       legend.labs = levels(Clinical_cluster$Clusters), 
                       xlab = "Survival time (Months)",  
                       legend.title = "", font.legend = 12, palette = c("#ac92eb", 
                                                                        "#4fc1e8",
                                                                        "#a0d568",
                                                                        "#ffce54",
                                                                        "#ed5564"))

png_op_os_path <- "OS.png"
png(png_op_os_path)
print(survp_os$plot, newpage = FALSE)
dev.off()

#======================================================================================
# For disease free survival - DFS
# Creating Survival Object
Surv_Obj_DFS <- Surv(Clinical_cluster$DFS_MONTHS,
                     Clinical_cluster$DFS_STATUS)

surv_diff <- survdiff(formula = Surv_Obj_DFS ~ Clinical_cluster$Clusters)
p_val <- (1 - pchisq(surv_diff$chisq, length(surv_diff$n) - 1))

fit <- do.call(survfit, list(formula = Surv_Obj_DFS ~ Clinical_cluster$Clusters, data = Clinical_cluster))

title = "Disease free survival"
survp_dfs <- ggsurvplot(fit, legend = c(0.8,0.8), pval = TRUE, title = title,
                        surv.median.line="hv", 
                        surv.scale = "percent", ylab = "Surviving",  
                        legend.labs = levels(Clinical_cluster$Clusters), 
                        xlab = "Survival time (Months)",  
                        legend.title = "", font.legend = 12, palette = c("#ac92eb", 
                                                                         "#4fc1e8",
                                                                         "#a0d568",
                                                                         "#ffce54",
                                                                         "#ed5564"))

png_op_dfs_path <- "DFS.png"
png(png_op_dfs_path)
print(survp_dfs$plot, newpage = FALSE)
dev.off()
