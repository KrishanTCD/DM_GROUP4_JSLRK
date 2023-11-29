# Load the installed packages
library(forecast)
library(readr)
library(ggplot2)
library(fpp2)
library(caret)
fraud.og <-read_csv("../card_transdata.csv")

# Exploration of dataset----
# Checking for size of the dataset
dim(fraud.og) # it has 1 million rows of observation and 8 columns
# check on col names
col_fraud <-colnames(fraud.og)
col_fraud
# Use case of each column available----

# Numeric
# distance_from_home: The distance from customers home to location of last transaction
# distance_from_last_transaction: Distance from last transaction to the last transaction
# These check are case the card is stolen and is being used in some other state.

# ratio_to_median_purchase_price: Ratio of purchased price transaction to median purchase price.
# Check in case user is suddenly doing a higher purchase than normal trend of the user.

# repeat_retailer: If the fraud occurred is it from same origin of purchase or seller.

# Binary
# used_chip: Chip implies if it was a credit card transaction.
# used_pin_number: Checking if the user pin was to see if the user pin was compromised.
# online_order: Major frauds occur online and as a result online fraud check is there
# fraud: Flag for fraudulent model yes or no

# Check for missing values, data type, data summary, unique values

table(fraud.og$fraud)

for(i in col_fraud){
  print(sprintf(" Class for col %s is %s",i,class(fraud.og[[i]])))
  }
sum(is.na(fraud.og))

# checking for correlation

heatmap(cor(fraud.og))

# Fraud and Ratio to Mean are highly correlated
summary(fraud.og)

# It is clear from here that there are values that are outliers to the common value in all the
# numerical columns which are of interest to us, thereby checking with boxplots

# Checking for outliers in given predictors
boxplot(fraud.og$distance_from_home,main="Checking for outliers in Distance from home check")
boxplot(fraud.og$distance_from_last_transaction,main="Checking for outliers in Distance from last tranx check")
boxplot(fraud.og$ratio_to_median_purchase_price,main="Checking for outliers in Distance from home check")
# It is clear that a lot of values are extending beyond 3rd quartile hence a lot of outliers

#Normalised values dataframe copy----
fraud.sc <- fraud.og
fraud.sc$distance_from_home<-scale(fraud.sc$distance_from_home)
fraud.sc$distance_from_last_transaction<-scale(fraud.sc$distance_from_last_transaction)
fraud.sc$ratio_to_median_purchase_price<-scale(fraud.sc$ratio_to_median_purchase_price)

# Checking relations of variables to fraud

flt_val <-fraud.sc[fraud.sc$fraud==1,c(1:7)]

### Checking Fraud distribution with standard distributed values of all columns
# Function to calculate the length for each specified number of standard deviations
sd_dist <- function(data1, col1) {
  mean_value1 <- mean(data1[[col1]])
  sd_threshold1 <- sd(data1[[col1]])
 #sd_pos1 meaning 1 positive standard deviation from mean and sd_neg1 meaning negative
  sd_2_to_infn <- length(data1[[ col1 ]][data1[[col1]] > mean_value1+(2*sd_threshold1)])
  sd_1_to_2 <- length(data1[[col1]][(data1[[col1]] < mean_value1+(2*sd_threshold1)) & (data1[[col1]] > mean_value1+(1*sd_threshold1))])
  mean_to_1 <- length(data1[[col1]][(data1[[col1]] < mean_value1+(1*sd_threshold1)) & (data1[[col1]] > mean_value1)])
  mean_to_neg1 <- length(data1[[col1]][(data1[[col1]] < mean_value1) & (data1[[col1]] > mean_value1-(1*sd_threshold1))])
  sd_1_to_neg2 <- length(data1[[col1]][(data1[[col1]] < mean_value1-(1*sd_threshold1)) & (data1[[col1]] > mean_value1-(2*sd_threshold1))])
  sd_1_to_neg3 <- length(data1[[col1]][(data1[[col1]] < mean_value1-(2*sd_threshold1))])

  return(c(sd_2_to_infn,sd_1_to_2,mean_to_1,mean_to_neg1,sd_1_to_neg2,sd_1_to_neg3))
  }
std_dist1 <-sd_dist(flt_val,"distance_from_home")
std_dist2 <-sd_dist(flt_val,"distance_from_last_transaction")
std_dist3 <-sd_dist(flt_val,"ratio_to_median_purchase_price")

result_dist <- as.data.frame(rbind(std_dist1,std_dist2,std_dist3),row.names=colnames(flt_val)[1:3]) 
colnames(result_dist)<- c("sd2 to +inf "," sd1 to sd2 "," mean to sd 1 "," sd-1 to mean "," sd-2 to sd-1 "," sd2 to -inf ")
result_dist

# It can be observed from the result_dist framework that majority of the frauds occur for mean -1sd to mean+1sd
# There are 2647 outliers from distance from home, 2232 from distance_from_last_transaction,, 3012 from ratio to median purchase price
# These outliers being marked fraud makes sense in business context as well but we are interested in catching those that are majority

# Check for number of fraud cases marked----

table(fraud.og$fraud) #912597 cases of no fraud and 87403 cases of fraud

# Using naive rule the accuracy of a new case to be not a fraud is very high
print(sprintf(" The Accuracy by Naive bias for no fraud is %g%%",100*round(sum(fraud.sc$fraud==0)/length(fraud.sc$fraud),5)))
print(sprintf(" The Accuracy by Naive bias for fraud is %g%%",100*round(sum(fraud.sc$fraud==1)/length(fraud.sc$fraud),5)))

# There is definitely a very high bias based on above distribution
barplot(table(fraud.sc$fraud), col=c("grey", "red"), main="Imbalance between fraud occurances", names.arg=c("0", "1"))
text(x=1,y=900000,labels=table(fraud.sc$fraud)[1],pos=2, col="black")
text(x=2,y=70000,labels=table(fraud.sc$fraud)[2],pos=2, col="black")


# To assess bias in the data, will use sampling 
# When the population can be divided into subgroups or strata that share similar characteristics. 
# This method ensures representation from each stratum and can lead to more precise and reliable results when analyzing subgroup differences.
install.packages("sampler")
library(sampler)
library(dplyr)

#take a 10% value to train models 
split.index <- createDataPartition(fraud.sc$fraud, p =0.1, list = FALSE)
fraud.sc <- fraud.sc[split.index,]
prop.table(table(fraud.sc$fraud))

# Creating Training Validation Split----
set.seed(12)

# Train Validation split using stratified sampling due to class imbalance
# The list has been set to false as we need the output as a vector index instead of a list of indexes
split_index <- createDataPartition(fraud.sc$fraud, p =0.7, list = FALSE)
fraud.train<-fraud.sc[split_index,]
fraud.valid<-fraud.sc[-split_index,]
table(fraud.sc$fraud)
table(fraud.train$fraud)

# Training data check and naive accuracy
length(fraud.train$fraud)
table(fraud.train$fraud)
print(sprintf(" Train: Naive: Accuracy NO FRAUD is %g%%",100*round(sum(fraud.train$fraud==0)/length(fraud.train$fraud),5)))
print(sprintf(" Train: Naive: Accuracy FRAUD is %g%%",100*round(sum(fraud.train$fraud==1)/length(fraud.train$fraud),5)))

# Testing data check and naive accuracy
length(fraud.valid$fraud)
table(fraud.valid$fraud)
print(sprintf(" Validation: Naive: Accuracy NO FRAUD is %g%%",100*round(sum(fraud.valid$fraud==0)/length(fraud.valid$fraud),5)))
print(sprintf(" Validation: Naive: Naive Accuracy FRAUD is %g%%",100*round(sum(fraud.valid$fraud==1)/length(fraud.valid$fraud),5)))

# Stratified sampling was success as accuracy for both train and test is equal implying that equal number of fraud cases are there in both

# Logistic regression" 

# Fit the model
fraud.glm <- glm(fraud ~., data = fraud.train, family = binomial)

# Summarize the model
summary(fraud.glm)

# Make predictions
probabilities.glm <- predict(fraud.glm, newdata = fraud.valid, type = "response")

# Predicted probabilities 
predicted.probs.df <- data.frame(Probability = probabilities.glm)

# Make predictions using a 0.25 threshold 
predicted.classes.glm <- ifelse(probabilities.glm > 0.25, 0, 1)

# Model accuracy
accuracy.glm <- mean(predicted.classes.glm == fraud.valid$fraud)
accuracy.glm
# Predicted class threshold is set at 0.25. conservative approach, classify more cases which are likely not fraud, on the safe side 
library(pROC) #for ROC curve
Roc.glm = roc(fraud.valid$fraud ~ probabilities.glm, plot = TRUE, print.auc = TRUE)
library(ggplot2)

# Create a density plot or histogram
# Alternatively, use geom_histogram() for a histogram:
# geom_histogram(aes(y = ..density..), fill = "blue", bins = 30, alpha = 0.5) +

ggplot(predicted.probs.df, aes(x = Probability)) +
  geom_density(fill = "blue", alpha = 0.5) +  # Density plot
  labs(title = "Distribution of Predicted Probabilities",
       x = "Predicted Probability",
       y = "Density") 
#geom_vline(xintercept = 0.4, color = "red",  size = 1)

# Decision tree 
library(rpart)
library(rpart.plot)
fraud.dt <- rpart(fraud~., data = fraud.train, method='class', cp = 0.0005, maxdepth=8, model = TRUE)
prp(fraud.dt)

# Prune the tree
printcp(fraud.dt)
plotcp(fraud.dt)
prunefit<-prune(fraud.dt,cp=fraud.dt$cptable[which.min(fraud.dt$cptable[,'xerror']),'CP'])
prp(prunefit)

prunefit<-rpart(fraud.dt, cp=0.001)
prp(prunefit)

# Assess accuracy 
predictions.dt <- predict(prunefit, newdata = fraud.valid, type = "class")
confusionmatrix.dt <- confusionMatrix(as.factor(predictions.dt), as.factor(fraud.valid$fraud), positive="1")

# Random forest 
install.packages("randomForest")
library(randomForest)
fraud.rf <- randomForest(fraud ~.,
                         data = fraud.train, ntree = 1000)
importance(fraud.rf)           
varImpPlot(fraud.rf)        
prob.random <- predict(fraud.rf, fraud.valid, type= "class")
pred.random <- ifelse(pred.random > 0.25, 0, 1)
confusionMatrix(as.factor(pred.random), as.factor(fraud.valid$fraud))

# Imbalanced Data Models----
library(rpart)
library(rpart.plot)
fraud.dt <- rpart(fraud~.,data=fraud.train,method="class")
prp(fraud.dt,yesno=2,box.palette = "GnRd",split.border.col = c("Black"),uniform=TRUE,main="Decision Tree without Pruning")
#detach(fraud.dt.train)
#Check cross validation data
fraud.dt$cptable
fraud.dt$variable.importance
fraud.dt.pruned<-prune(fraud.dt,cp=fraud.dt$cptable[which.min(fraud.dt$cptable[,'xerror']),'CP'])
fraud.dt.pruned$cptable
prp(fraud.dt.pruned,yesno=2,box.palette = "GnRd",split.border.col = c("Black"),uniform=TRUE,main="Decision Tree Pruned")
fraud.dt.pruned$variable.importance
rpart.rules(fraud.dt.pruned)

fraud.train.pred <- predict(fraud.dt.pruned, fraud.valid,type="class")
confusionMatrix(as.factor(fraud.train.pred),as.factor(fraud.valid$fraud))

fraud.valid.pred <-predict(fraud.dt.pruned, fraud.valid)
confusionMatrix(as.factor(fraud.valid.pred),fraud.valid$fraud,positive="1")

## Next models should be random forest, xgboost, bagging 
## use a ramdon forest: 

install.packages('randomForest') 
library(randomForest)


bestmtry <- tuneRF(fraud.train, fraud.train$fraud,stepFactor = 1.2, improve = 0.01, trace=T, plot= T) 
fraud.rf <- randomForest(fraud~.,data= fraud.train)
importance(fraud.rf) 
varImpPlot(fraud.rf)

pred.random.prob <- predict(fraud.rf, newdata = fraud.valid, type= "class")
pred.random<-ifelse(pred.random.prob>0.5,1,0)

table(pred.random,fraud.valid$fraud)
confusionMatrix(as.factor(pred.random),as.factor(fraud.valid$fraud))
Roc.random = roc(fraud.valid$fraud ~ pred.random.prob, plot = TRUE, print.auc = TRUE)



# KNN
# library(FNN)
# # Since k is selected using train and test, we split the test further to validation
# split_index_knn <- createDataPartition(fraud.valid$fraud, p =0.8, list = FALSE)
# fraud.knn.train<-fraud.train
# fraud.knn.valid<-fraud.valid[split_index_knn,]
# fraud.knn.test<-fraud.valid[-split_index_knn,]
# 
# accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))
#Error in knn(fraud.knn.train[, 1:7], fraud.knn.valid[, 1:7], cl = as.vector(fraud.knn.train[,  : 
# 'train' and 'class' have different lengths
# compute knn for different k on validation.
#
# for(i in 1:8) {
#   knn.pred <- knn(fraud.knn.train[,1:7],
#                   fraud.knn.valid[,1:7],
#                   cl = fraud.knn.train$fraud, k = i)
# 
#   accuracy.df[i, 2] <- confusionMatrix(factor(knn.pred>0.5, levels = c(0,1)),
#                                        factor(fraud.knn.valid$fraud>0.5,
#                                               levels = c(0,1)))$overall[1]
# }
# 
# # dim(fraud.knn.train[,1:7])
# # dim(fraud.knn.train[,8])
# 
# #Plot accuracy for knn for selecting best k
# plot(accuracy.df[,1], accuracy.df[,2], type = "l")
# 
# # Rebuild using the k value
# knn.pred <- knn(fraud.knn.train[,1:7], fraud.knn.test[,1:7],cl = fraud.knn.train[,8], k = i)

# Decision Tree
library(rpart)
library(rpart.plot)
# fraud.dt.train<-fraud.train
# fraud.dt.train$distance_from_home<-as.factor(fraud.dt.train$distance_from_home)
# fraud.dt.train$distance_from_last_transaction<-as.factor(fraud.dt.train$distance_from_last_transaction)
# fraud.dt.train$ratio_to_median_purchase_price<-as.factor(fraud.dt.train$ratio_to_median_purchase_price)

# fraud.dt.valid<-fraud.valid
# fraud.dt.valid$distance_from_home<-as.factor(fraud.dt.valid$distance_from_home)
# fraud.dt.valid$distance_from_last_transaction<-as.factor(fraud.dt.valid$distance_from_last_transaction)
# fraud.dt.valid$ratio_to_median_purchase_price<-as.factor(fraud.dt.valid$ratio_to_median_purchase_price)
#attach(fraud.dt.train)

