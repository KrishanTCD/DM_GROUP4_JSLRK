# #model using sampling SMOTE 
# 
# fraud.smote <-read_csv("/Users/sarahraubenheimer/Downloads/fraud_smote.csv")
# fraud.test <-read_csv("/Users/sarahraubenheimer/Downloads/fraud_test.csv")
# 
# table(fraud.smote$fraud)
# table(fraud.test$fraud)
# 
# #scale data 
# fraud.sc.smote <- fraud.smote
# fraud.sc.smote$distance_from_home<-scale(fraud.sc.smote$distance_from_home)
# fraud.sc.smote$distance_from_last_transaction<-scale(fraud.sc.smote$distance_from_last_transaction)
# fraud.sc.smote$ratio_to_median_purchase_price<-scale(fraud.sc.smote$ratio_to_median_purchase_price)
# 
# #Testing data
# fraud.sc.test <- fraud.test
# fraud.sc.test$distance_from_home<-scale(fraud.sc.test$distance_from_home)
# fraud.sc.test$distance_from_last_transaction<-scale(fraud.sc.test$distance_from_last_transaction)
# fraud.sc.test$ratio_to_median_purchase_price<-scale(fraud.sc.test$ratio_to_median_purchase_price)
# 
# split_index2 <- createDataPartition(fraud.sc.test$fraud, p =0.8, list = FALSE)
# fraud.sc.test1<-fraud.sc[split_index2,]
# table(fraud.sc.test1$fraud)
# fraud.sc.test2<-fraud.sc[-split_index2,]
# table(fraud.sc.test2$fraud)
# 
# #take 10% of date to run on 
# split_index <- createDataPartition(fraud.sc$fraud, p =0.9, list = FALSE)
# fraud.train<-fraud.sc[split_index,]
# fraud.valid<-fraud.sc[-split_index,]
# table(fraud.train$fraud)
# 
# library(randomForest)
# 
# 
# bestmtry <- tuneRF(fraud.train, fraud.train$fraud,stepFactor = 1.2, improve = 0.01, trace=T, plot= T) 
# fraud.rf <- randomForest(fraud~.,data= fraud.train)
# importance(fraud.rf) 
# varImpPlot(fraud.rf)
# 
# pred.random <- predict(fraud.rf, newdata = fraud.valid, type= "class")
# pred.random<-ifelse(pred.random>0.5,1,0)
# 
# table(pred.random,fraud.valid$fraud)
# confusionMatrix(as.factor(pred.random),as.factor(fraud.valid$fraud))
# 
# # With untouched data
# pred.random <- predict(fraud.rf, newdata = fraud.sc.test1, type= "class")
# pred.random<-ifelse(pred.random>0.5,1,0)
# 
# table(pred.random,fraud.sc.test1$fraud)
# confusionMatrix(as.factor(pred.random),as.factor(fraud.sc.test1$fraud))
# 
# 
# # Without the median ratio
# bestmtry2 <- tuneRF(fraud.train[,-c(3)], fraud.train[,-c(3)]$fraud,stepFactor = 1.2, improve = 0.01, trace=T, plot= T) 
# fraud.rf2 <- randomForest(fraud~.,data= fraud.train[,-c(3)])
# importance(fraud.rf2) 
# varImpPlot(fraud.rf2)
# 
# pred.random2 <- predict(fraud.rf2, newdata = fraud.valid[,-c(3)], type= "class")
# pred.random2<-ifelse(pred.random2>0.5,1,0)
# 
# table(pred.random2,fraud.valid[,-c(3)]$fraud)
# confusionMatrix(as.factor(pred.random2),as.factor(fraud.valid[,-c(3)]$fraud))

#run smote data: 
fraud.smote <-read_csv("/Users/sarahraubenheimer/Downloads/fraud_smote.csv")
fraud.test <-read_csv("/Users/sarahraubenheimer/Downloads/fraud_test.csv")

table(fraud.smote$fraud)
table(fraud.test$fraud)

#scale data 
fraud.sc.smote <- fraud.smote
fraud.sc.smote$distance_from_home<-scale(fraud.sc.smote$distance_from_home)
fraud.sc.smote$distance_from_last_transaction<-scale(fraud.sc.smote$distance_from_last_transaction)
fraud.sc.smote$ratio_to_median_purchase_price<-scale(fraud.sc.smote$ratio_to_median_purchase_price)

split_index2 <- createDataPartition(fraud.sc.smote$fraud, p =0.8, list = FALSE)
fraud.smote.train1<-fraud.sc[split_index2,]
prop.table(table(fraud.smote.train1$fraud))
fraud.smote.train2<-fraud.sc[-split_index2,]
prop.table(table(fraud.smote.train2$fraud))

library(randomForest)


bestmtrysmote <- tuneRF(fraud.smote.train1, fraud.smote.train1$fraud,stepFactor = 1.2, improve = 0.01, trace=T, plot= T) 
fraud.rf.smote <- randomForest(fraud~.,data= fraud.smote.train1)
importance(fraud.rf.smote) 
varImpPlot(fraud.rf.smote)

pred.random.smote <- predict(fraud.rf.smote, newdata = fraud.smote.train2, type= "class")
pred.random.smote<-ifelse(pred.random>0.5,1,0)

table(pred.random.smote,fraud.smote.train2$fraud)
confusionMatrix(as.factor(pred.random.smote),as.factor(fraud.sc.test2$fraud))

# With untouched data
pred.random.smote2 <- predict(fraud.rf.smote, newdata = fraud.valid, type= "class")
pred.random.smote2<-ifelse(pred.random.smote2>0.5,1,0)

table(pred.random.smote2,fraud.valid$fraud)
confusionMatrix(as.factor(pred.random.smote2),as.factor(fraud.valid$fraud))

#run regression 

fraud.glm.smote <- glm(fraud ~., data = fraud.smote.train1, family = binomial)
# Summarize the model
summary(fraud.glm.smote)

# Make predictions
probabilities.glm.smote <- predict(fraud.glm.smote, newdata = fraud.smote.train2, type = "response")
#predicted probabilities 
predicted.probs.df.smote <- data.frame(Probability = probabilities.glm.smote)
#make predictions using a 0.5 threshold 
predicted.classes.glm.smote <- ifelse(probabilities.glm.smote > 0.5, 1, 0)
# Model accuracy
accuracy.glm.smote <- mean(predicted.classes.glm.smote == fraud.smote.train2$fraud)

accuracy.glm
#predicted class threshold is set at 0.4. conservative approach, classify more cases which are likely not fraud, on the safe side 

table(predicted.classes.glm.smote,fraud.smote.train2$fraud)
confusionMatrix(as.factor(predicted.classes.glm.smote),as.factor(fraud.smote.train2$fraud))


#run on unseen data 
probabilities.glm.smote2 <- predict(fraud.glm.smote, newdata = fraud.valid, type = "response")
#predicted probabilities 
predicted.probs.df.smote2 <- data.frame(Probability = probabilities.glm.smote2)
#make predictions using a 0.5 threshold 
predicted.classes.glm.smote2 <- ifelse(probabilities.glm.smote > 0.5, 1, 0)
# Model accuracy
accuracy.glm.smote2 <- mean(predicted.classes.glm.smote2 == fraud.valid$fraud)

accuracy.glm
#predicted class threshold is set at 0.4. conservative approach, classify more cases which are likely not fraud, on the safe side 
table(predicted.classes.glm.smote2,fraud.valid$fraud)
confusionMatrix(as.factor(predicted.classes.glm.smote2),as.factor(fraud.valid$fraud))

# library(pROC)
# 
# Roc.glm = roc(fraud.valid$fraud ~ probabilities.glm, plot = TRUE, print.auc = TRUE)
# 
# 
# library(ggplot2)
# # Create a density plot or histogram
# ggplot(predicted.probs.df, aes(x = Probability)) +
#   geom_density(fill = "blue", alpha = 0.5) +  # Density plot
#   # Alternatively, use geom_histogram() for a histogram:
#   # geom_histogram(aes(y = ..density..), fill = "blue", bins = 30, alpha = 0.5) +
#   
#   labs(title = "Distribution of Predicted Probabilities",
#        x = "Predicted Probability",
#        y = "Density") 
# #geom_vline(xintercept = 0.4, color = "red",  size = 1)


#decision tree 
library(rpart)
library(rpart.plot)
fraud.dt.smote <- rpart(fraud~.,data= fraud.smote.train1, method="class")
prp(fraud.dt.smote,yesno=2,box.palette = "GnRd",split.border.col = c("Black"),uniform=TRUE,main="Decision Tree without Pruning")
#detach(fraud.dt.train)
#Check cross validation data
fraud.dt.smote$cptable
fraud.dt.smote$variable.importance
fraud.dt.pruned.smote<-prune(fraud.dt.smote,cp=fraud.dt$cptable[which.min(fraud.dt$cptable[,'xerror']),'CP'])
fraud.dt.pruned.smote$cptable
prp(fraud.dt.pruned.smote,yesno=2,box.palette = "GnRd",split.border.col = c("Black"),uniform=TRUE,main="Decision Tree Pruned")
fraud.dt.pruned.smote$variable.importance
rpart.rules(fraud.dt.pruned.smote)

fraud.train.pred.smote <- predict(fraud.dt.pruned.smote, fraud.smote.train2,type="class")
confusionMatrix(as.factor(fraud.train.pred.smote),as.factor(fraud.smote.train2$fraud))

#predict on unseen data;

fraud.train.pred.smote2 <- predict(fraud.dt.pruned.smote, fraud.valid,type="class")
confusionMatrix(as.factor(fraud.train.pred.smote2),as.factor(fraud.valid$fraud))

#boosted tree

set.seed(24)
fraud.boost.smote<- gbm(fraud ~ . , distribution = "bernoulli", 
                  data=fraud.smote.train1, n.trees = 500, interaction.depth=4,
                  shrinkage = 0.1) # learning rate

summary(fraud.boos.smote)
pred.boost.probability.smote<-predict(fraud.boost.smote,newdata=fraud.smote.train2,n.trees=5000,type="response")
pred.boost.smote<-ifelse(pred.boost.probability.smote>0.5,1,0)

table(pred.boost.smote,fraud.smote.train2$fraud)
confusionMatrix(as.factor(pred.boost.smote),as.factor(fraud.smote.train2$fraud))


#run on unseen data: 
pred.boost.probability.smote2<-predict(fraud.boost.smote,newdata=fraud.valid,n.trees=5000,type="response")
pred.boost.smote2<-ifelse(pred.boost.probability.smote2 >0.5,1,0)

table(pred.boost.smote2,fraud.valid$fraud)
confusionMatrix(as.factor(pred.boost.smote2),as.factor(fraud.valid$fraud))

Roc.random = roc(fraud.valid$fraud ~ pred.random.prob, plot = TRUE, print.auc = TRUE)












