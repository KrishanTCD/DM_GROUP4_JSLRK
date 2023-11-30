#use random sample of data taking 10% of the arguments = 10 000 
library(dplyr)
fraud.og <-read_csv("card_transdata.csv")
set.seed(123)  
shuffle_index <- sample(1:nrow(fraud.og))
head(shuffle_index)
fraud.og <- fraud.og[shuffle_index, ]
head(fraud.og)

# Sampling only 100000 from 1 mil using random sampling for computational purpose
fraud.sample <- fraud.og %>% sample_n(100000)
prop.table(table(fraud.sample$fraud))
prop.table(table(fraud.og$fraud))
set.seed(12)

# Stratified sampling for Train Test Split
# Train Validation split using stratified sampling due to class imbalance
# The list has been set to false as we need the output as a vector index instead of a list of indexes
split_index <- createDataPartition(fraud.sample$fraud, p =0.8, list = FALSE)
fraud.train<-fraud.sample[split_index,]
fraud.valid<-fraud.sample[-split_index,]
table(prop.table(fraud.train$fraud))
table(prop.table(fraud.valid$fraud))

# Logistic Regression model
fraud.glm <- glm(fraud ~., data = fraud.train, family = binomial)

# Summarize the model
summary(fraud.glm)
accuracy(fraud.glm)

# Make predictions
probabilities.glm <- predict(fraud.glm, newdata = fraud.valid, type = "response")

#predicted probabilities 
predicted.probs.df <- data.frame(Probability = probabilities.glm)

# make predictions using a 0.5 threshold 
predicted.classes.glm <- ifelse(probabilities.glm > 0.5, 1, 0)

# Model accuracy
accuracy.glm <- mean(predicted.classes.glm == fraud.valid$fraud)
accuracy.glm

# Predicted class threshold is set at 0.5. conservative approach, classify more cases which are likely not fraud, on the safe side 
# regresion model gets accuracy of 0.95645
library(pROC) # For ROC plot

#Plotting ROC
Roc.glm = roc(fraud.valid$fraud ~ probabilities.glm, plot = TRUE, print.auc = TRUE)
library(ggplot2)
# Create a density plot or histogram
ggplot(predicted.probs.df, aes(x = Probability)) +
  geom_density(fill = "blue", alpha = 0.5) +  # Density plot
  # Alternatively, use geom_histogram() for a histogram:
  # geom_histogram(aes(y = ..density..), fill = "blue", bins = 30, alpha = 0.5) +
  labs(title = "Distribution of Predicted Probabilities",
       x = "Predicted Probability",
       y = "Density") 
#geom_vline(xintercept = 0.4, color = "red",  size = 1)

# Decisoin Tree 
library(rpart)
library(rpart.plot)
fraud.dt <- rpart(fraud~.,data=fraud.train,method="class")
prp(fraud.dt,yesno=2,box.palette = "GnRd",split.border.col = c("Black"),uniform=TRUE,main="Decision Tree without Pruning")
#detach(fraud.dt.train)

# Check cross validation data
fraud.dt$cptable
fraud.dt$variable.importance

# Pruning the Tree
fraud.dt.pruned<-prune(fraud.dt,cp=fraud.dt$cptable[which.min(fraud.dt$cptable[,'xerror']),'CP'])
fraud.dt.pruned$cptable
prp(fraud.dt.pruned,yesno=2,box.palette = "GnRd",split.border.col = c("Black"),uniform=TRUE,main="Decision Tree Pruned")
fraud.dt.pruned$variable.importance
rpart.rules(fraud.dt.pruned)

# Training Accuracy
fraud.train.pred <- predict(fraud.dt.pruned, fraud.train, type="class")
confusionMatrix(as.factor(fraud.train.pred),as.factor(fraud.train$fraud))

# Validation Accuracy
fraud.valid.pred <-predict(fraud.dt.pruned, fraud.valid, type="class")
confusionMatrix(as.factor(fraud.valid.pred),as.factor(fraud.valid$fraud))

# Checking distribution on each node
# Command nn gives index of each node and leaf
prp(fraud.dt.pruned,yesno=2,box.palette = "GnRd",nn=TRUE,split.border.col = c("Black"),uniform=TRUE,main="Decision Tree Pruned")
fraud.dt.pruned$frame[,1:2]

# Random forest on sample data 
install.packages('randomForest') 
library(randomForest)

# For tuning RF model
bestmtry <- tuneRF(fraud.train, fraud.train$fraud,stepFactor = 1.2, improve = 0.01, trace=T, plot= T) 
fraud.rf <- randomForest(fraud~.,data= fraud.train)
importance(fraud.rf) 
varImpPlot(fraud.rf)

pred.random.prob <- predict(fraud.rf, newdata = fraud.valid, type= "class")
pred.random<-ifelse(pred.random.prob>0.5,1,0)

table(pred.random,fraud.valid$fraud)
confusionMatrix(as.factor(pred.random),as.factor(fraud.valid$fraud))
Roc.random = roc(fraud.valid$fraud ~ pred.random.prob, plot = TRUE, print.auc = TRUE)

# Gradient boosted Method Tree
install.packages('gbm')
library(gbm)
library(MASS)

# Validation Set 
# Build the Boosted Regression Model
set.seed(24)
fraud.boost<- gbm(fraud ~ . , distribution = "bernoulli", 
                    data=fraud.train, n.trees = 500, interaction.depth=4,
                    shrinkage = 0.1) # learning rate

summary(fraud.boost)
pred.boost.probability<-predict(fraud.boost,newdata=fraud.valid,n.trees=5000,type="response")
pred.boost<-ifelse(pred.boost.probability>0.5,1,0)

table(pred.boost,fraud.valid$fraud)
confusionMatrix(as.factor(pred.boost),as.factor(fraud.valid$fraud))
Roc.random = roc(fraud.valid$fraud ~ pred.random.prob, plot = TRUE, print.auc = TRUE)
