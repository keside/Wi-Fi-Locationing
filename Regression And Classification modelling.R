
# Library loadings
library(tidyverse)
library(MLmetrics)
library(caret)
library(kknn)
library(randomForest)
library(C50)
library(e1071)
library(mlbench)
library(ggplot2)
library(scatterplot3d)
# library(DMwR) # compute all the error metrics in regression


# Saving my new dataset ----

write.csv(new_trainingSet, 'New_trainSet.csv', row.names = F)

write.csv(new_validationSet, 'New_validSet.csv', row.names = F)


#Loading New Datasets

new_trainingSet <- read.csv('data/preprocessed/New_trainSet.csv',header = T, sep = ',')
new_validationSet <- read.csv('data/preprocessed/New_validSet.csv', header = T, sep = ',')


dim(new_validationSet)
dim(new_trainingSet)

#EDA
str(new_trainingSet)
str(new_validationSet)
table(new_trainingSet$FL)

new_trainingSet$FL <- as.factor(new_trainingSet$FL)

new_validationSet$FL <- as.factor(new_validationSet$FL)

new_trainingSet$BU <- as.factor(new_trainingSet$BU)

new_validationSet$BU <- as.factor(new_validationSet$BU)

#### Rename the floor and Building variable for easily identification

new_trainingSet$FL <- recode(new_trainingSet$FL, '0'=1, '1'=2, '2'=3, '3'=4, '4'=5)

new_trainingSet$FL <- recode(new_trainingSet$FL, '1'= 'A', '2'='B', '3'= 'C', '4'='D', '5'='E')


# I will remove classes that is not needed anymore.

new_trainingSet$RP <- NULL
new_trainingSet$SP <- NULL

#
new_validationSet$RP <- NULL
new_validationSet$SP <- NULL

#####

# First I will split the data for each different features.

c <- as.numeric(ncol(new_trainingSet))

Classes <- new_trainingSet[,c(1:c)]

#Classes & Features
LO <- as.data.frame(new_trainingSet[,-c(95,96,97)])

LA <- as.data.frame(new_trainingSet[,-c(94,96,97)])

FL <- as.data.frame(new_trainingSet[,-c(94,95,97)])

BU <- as.data.frame(new_trainingSet[,-c(94,95,96)])



# Validation DataSet


LO1 <- new_validationSet[,-c(95,96,97)]
LO1 <- as.data.frame(LO1)
LA1 <- new_validationSet[,-c(94,96,97)]
LA1 <- as.data.frame(LA1)
FL1 <- new_validationSet[,-c(94,95,97)]
FL1 <- as.data.frame(FL1)
BU1 <- as.data.frame(new_validationSet[,-c(94,95,96)])


# Create Partition

# FLOOR ## Training Dataset

set.seed(601)

trainIndex_FL <- createDataPartition(y = FL$FL,
                                   p = .70,
                                   list = FALSE)


FLTrain <- as.data.frame(FL[trainIndex_FL,])
FLTest <-  as.data.frame(FL[-trainIndex_FL,])


# FLOOR ## Validation Dataset

trainIndex_FL1 <- createDataPartition(y = FL1$FL,
                                   p = .70,
                                   list = FALSE)

FL1Train <- as.data.frame(FL1[trainIndex_FL1,])
FL1Test <-  as.data.frame(FL1[-trainIndex_FL1,])

#Longitude - create data partition####
set.seed(601)
trainIndex5 <- createDataPartition(y = LO$LO,
                                   p = .70,
                                   list = FALSE)
LOTrain <- as.data.frame(LO[trainIndex5,])
LOTest <-  as.data.frame(LO[-trainIndex5,])


# Longitude ## Validation Dataset
trainIndex_VL <- createDataPartition(y = LO1$LO,
                                   p = .70,
                                   list = FALSE)
LO1Train <- as.data.frame(LO1[trainIndex_VL,])
LO1Test <-  as.data.frame(LO1[-trainIndex_VL,])



#Latitude - create data partition ####
set.seed(601)
trainIndex6 <- createDataPartition(y = LA$LA,
                                   p = .70,
                                   list = FALSE)
LATrain <- as.data.frame(LA[trainIndex6,])
LATest <-  as.data.frame(LA[-trainIndex6,])

# Latitude ## Validation Dataset
trainIndex_LA <- createDataPartition(y = LA1$LA,
                                   p = .70,
                                   list = FALSE)
LA1Train <- as.data.frame(LA1[trainIndex_LA,])
LA1Test <-  as.data.frame(LA1[-trainIndex_LA,])


# Building - create data partition

# Training Dataset
set.seed(601)
trainIndex_BU_T <- createDataPartition(y = BU$BU,
                                     p = .70,
                                     list = F)
BUTrain <- as.data.frame(BU[trainIndex_BU_T,])
BUTest <-  as.data.frame(BU[-trainIndex_BU_T,])



# Validation Dataset
set.seed(601)
trainIndex_BU <- createDataPartition(y = BU1$BU,
                                     p = .70,
                                     list = F)
BU1Train <- as.data.frame(BU1[trainIndex_BU,])
BU1Test <-  as.data.frame(BU1[-trainIndex_BU,])


#Control for Training

#Create Control Cross Validation

ctrl <- trainControl(method = "repeatedcv",
                     repeats = 3,
                     summaryFunction = multiClassSummary)


ctrlk <- trainControl(method = "cv",
                      number = 10,
                      summaryFunction = multiClassSummary,
                      classProbs = TRUE,
                      allowParallel = T)

ctrl1 <- trainControl(method = "repeatedcv",
                      repeats = 3,
                      classProbs = TRUE)

ctrl2 <- trainControl(method = "repeatedcv",
                      summaryFunction = twoClassSummary,
                      repeats = 3,
                      classProbs = TRUE)










# Modelling ----

# LONGITUDE USING WEIGHTED K NEAREST NEIGHOUR 

Train <- as.data.frame(LOTrain[,])
Test <- as.data.frame(LOTest[,])

?train.kknn

kknn_LO <- train.kknn(LO~.,
                        data = Train,
                        trControl = ctrl,
                        method = "kknn",
                        method = "optimal",
                        kmax = 10)
#
plot(kknn_LO)


print(kknn_LO)


#prediction
kknn_LO_predic <- predict(kknn_LO,Test)

print(kknn_LO_predic)

barchart(kknn_LO_predic)

## Calculating prediction accuracy and error rates
# Firstly make actuals_predicted a dataframe.

actuals_preds <- data.frame(cbind(actuals=Test$LO, predicted=kknn_LO_predic))  

head(actuals_preds)
# actuals predicted
# 1 -7519.152 -7523.219
# 2 -7528.816 -7528.425
# 3 -7537.340 -7537.428
# 4 -7512.604 -7510.020
# 5 -7501.222 -7501.867
# 6 -7474.653 -7475.645

#
scatter.smooth(actuals_preds)

hist(actuals_preds)

#  actuals and predicted values have similar directional movement ie Higher corr accuracy
correlation_accuracy <- cor(actuals_preds)  # 100% 

barchart(correlation_accuracy)

## Error Rates

error_rate <- DMwR::regr.eval(actuals_preds$actuals, actuals_preds$predicted) 
#      mae            mse            rmse           mape 
#  3.833418e+00  8.396016e+01   9.162978e+00      5.152668e-04

summary(error_rate)

barplot(error_rate)

##
# Min-Max Accuracy Calculation
min_max_accuracy <- mean(min(actuals_preds$actuals,actuals_preds$predicted )/
                           max(actuals_preds$actuals,actuals_preds$predicted ))  # 100% ie Higher the better

# MAPE Calculation
mape <- mean(abs((actuals_preds$predicted - actuals_preds$actuals))/actuals_preds$actuals)  
# => -51.52%, ie 0  # The lower the better model


# Validation Dataset ## LO1
TrainLO1 <- as.data.frame(LO1Train[,])
TestL01 <- as.data.frame(LO1Test[,])


kknn_LO1 <- train.kknn(LO~.,
                      data = TrainLO1,
                      trControl = ctrl,
                      method = "kknn",
                      method = "optimal",
                      kmax = 10)
##

plot(kknn_LO1)

## 
print(kknn_LO1) # Best Optimal Kernel is 3


#prediction
kknn_LO1_predic <- predict(kknn_LO1, TestL01)

plot(kknn_LO1_predic)

## Calculating prediction accuracy and error rates
# Firstly make actuals_predicted a dataframe.

actuals_preds_LO1 <- data.frame(cbind(actuals=TestL01$LO, predicted=kknn_LO1_predic)) 

head(actuals_preds_LO1)
#   actuals predicted
#1 -7515.917 -7562.443
#2 -7641.499 -7647.509
#3 -7345.085 -7350.682
#4 -7372.664 -7365.552
#5 -7377.068 -7367.060
#6 -7385.872 -7380.963

#
scatter.smooth(actuals_preds_LO1)

#
boxplot(actuals_preds_LO1)

# Correlation Accuracy and Error Rate
#  actuals and predicted values have similar directional movement ie Higher corr accuracy
correlation_accuracy_LO1 <- cor(actuals_preds_LO1)  # 100% 

#
barchart(correlation_accuracy_LO1)

## Error Rates

error_rate_LO1 <- DMwR::regr.eval(actuals_preds_LO1$actuals, actuals_preds_LO1$predicted) 

print(error_rate_LO1)
#     mae          mse         rmse         mape 
#  1.031506e+01 5.340206e+02 2.310889e+01 1.379056e-03

barplot(error_rate_LO1/100)


# Longitude - Random Forest ----

train <- as.data.frame(Train[,])
test <- as.data.frame(Test[,])


rf_LO <- randomForest(LO~.,
                        data = train,
                        trControl = ctrl,
                        importance = TRUE,
                        ntree= 500,
                        maximize =TRUE)
#
plot(rf_LO)

#
summary(rf_LO)

#prediction with the modelo####
rf_LO_predic <- predict(rf_LO, test)

## Calculating prediction Corr accuracy and error rates
actuals_preds_RF <- data.frame(cbind(actuals=test$LO, predicted=rf_LO_predic)) 

head(actuals_preds_RF)

scatter.smooth(actuals_preds_RF)

## Corr Accuracy
correlation_accuracy_RF <- cor(actuals_preds_RF)  # 100%

#
barchart(correlation_accuracy_LO1)

## Error Rates

error_rate_RFLO <- DMwR::regr.eval(actuals_preds_RF$actuals, actuals_preds_RF$predicted) 


print(error_rate_RFLO)
#     mae           mse         rmse          mape 
# 4.890699e+00 8.965373e+01 9.468565e+00 6.577825e-04 

barplot(error_rate_RFLO/100)



# Validation Dataset ## LO1

rf_LO1 <- randomForest(LO~.,
                      data = LO1Train,
                      trControl = ctrl,
                      importance = TRUE,
                      ntree= 500)
plot(rf_LO1)                      

#prediction with the modelo####
rf_LO1_predic <- predict(rf_LO1, LO1Test)

plot(rf_LO1_predic)

# Calculating prediction Corr accuracy and error rates
actuals_preds_RFLO1 <- data.frame(cbind(actuals=LO1Test$LO, predicted=rf_LO1_predic)) 

head(actuals_preds_RFLO1)

scatter.smooth(actuals_preds_RFLO1)

## Corr Accuracy
correlation_accuracy_RFLO1 <- cor(actuals_preds_RFLO1)


#
barchart(correlation_accuracy_RFLO1)


## Error Rates
error_rate_RFLO1 <- DMwR::regr.eval(actuals_preds_RFLO1$actuals, actuals_preds_RFLO1$predicted) 


print(error_rate_RFLO1)
#  mae          mse         rmse         mape 
# 1.083157e+01 2.363144e+02 1.537252e+01 1.445898e-03 


barplot(error_rate_RFLO1/100) # mse > rmse > mae > mape





  
# LATITUDE - Nearest neighbour ---- 
train <- as.data.frame(LATrain[,])
test <- as.data.frame(LATest[,])


set.seed(601)

tkkn_LA <- system.time(
  kknn_LA <- train.kknn(LA~.,
                        data = train,
                        trControl = ctrl,
                        method = "kknn",
                        method = "optimal",
                        kmax = 5)
)

# prediction 
kknn_LA_predic <- predict(kknn_LA, LATest)

#

plot(kknn_LA)


# Latitude ## Validation Dataset
train_VA <- as.data.frame(LA1Train[,])
test_VA <- as.data.frame(LA1Test[,])


set.seed(601)

tkkn_LA1 <- system.time(
  kknn_LA1 <- train.kknn(LA~.,
                        data = train_VA,
                        trControl = ctrl,
                        method = "kknn",
                        method = "optimal",
                        kmax = 5)
)

# prediction 
kknn_LA1_predic <- predict(kknn_LA1, test_VA)

#
plot(kknn_LA1)



# Latitude - Random Forest
set.seed(601)

trf_LA <- system.time(
  rf_LA <- randomForest(LA~.,
                        data = train,
                        trControl = ctrl,
                        importance = TRUE,
                        ntree= 100,
                        maximize =TRUE
  )
)

# prediction
rf_LA_predic <- predict(rf_LA, LATest, level = .95)

#
rf_LA

# Latitude - SVM
set.seed(601)
tsvm_LA <- system.time(
  svm_LA <- svm(LA~.,
                data = train,
                trControl = ctrl,
                preProc = c("center", "scale")
  )
)


#prediction with the model 
svm_LA_predic <- predict(svm_LA, LATest, level = .95)

# 
svm_LA






## ALTITUDE ----

# This is a classification problem.

trainFL <- as.data.frame(FLTrain[,])
testFL <- as.data.frame(FLTest[,])

str(trainFL)

## Floor - Random Forest
metric <- "Accuracy"

#
set.seed(601)

RF_FL <- randomForest(FL~., data= trainFL, trControl = ctrl, tune.randomForest='best.foo')

print(RF_FL)


# Prediction and Confusion Matrix -- Train 

P1 <- predict(RF_FL, trainFL)

barchart(P1)

confusionMatrix(P1, trainFL$FL)

tab_FL <- table(trainFL$FL, P1)

# Misclassification Error

Err_TrainFL <- 1-sum(diag(tab_FL))/sum(tab_FL) # 0.001934



# prediction with the model -- Test
P2 <- predict(RF_FL, testFL, level=.95)

plot(P2) # D > B > A > C > E # Classes with highest wifi location

confusionMatrix(P2, testFL$FL) # Accuracy : 0.9981 and Kappa : 0.9975

tab_Rf <- table(testFL$FL, P2)

# Misclassification Error

Err_TestFL <- 1-sum(diag(tab_Rf))/sum(tab_Rf) # 0.0307

# OOB Error
Err_TrainFL + Err_TestFL # 3.3%
 

# Error rate of RF
plot(RF_FL)

# No. of nodes for the trees
hist(treesize(RF_FL), main ='No of Nodes for the Trees', col ='green')


# Variable Importance
varImpPlot(RF_FL,
           sort = T, n.var = 10, main = 'Top 10 -Variable Importances')
# PC4, PC17, PC5, PC13, PC10, PC3, PC9, PC19, PC14, PC22

importance(RF_FL)

varUsed(RF_FL)

# Partial Dependence Plot
partialPlot(RF_FL, testFL, PC1, "E")


#Extract Single tree
getTree(RF_FL, 1, labelVar = T)

getT

# Multi-demensional Scaling Plot of Proximity Matrix
# MDSplot(RF_FL, testFL$FL)





# Floor - C50
trainFL <- as.data.frame(trainFL)
testFL<- as.data.frame(testFL)


c50_FL <- C5.0(FL~.,data = trainFL,
                        trControl = c(ctrl1, noGlobalPruning = TRUE))

print(c50_FL)



#prediction - Test dataset 
c50_FL_predic <- predict(c50_FL, testFL, level = .95)

barchart(c50_FL_predic) # # D > B > A > C > E # Classes with highest wifi location


confusionMatrix(testFL$FL, c50_FL_predic) # Accuracy : 0.8976 and 0.8675


# Misclassification Error
tab_c50 <- table(Actual= testFL$FL, predicted = c50_FL_predic)

1-sum(diag(tab_c50))/sum(tab_c50) # 0.102

# Accuracy
sum(diag(tab_c50))/sum(tab_c50) # 0.8976246





# Floor - SVM
trainFL <- as.data.frame(trainFL)
testFL<- as.data.frame(testFL)

mymodel <- svm(FL~., trainFL, kernel= 'radial', method = 'ctrlk')

print(mymodel)

summary(mymodel)


#Confusion Matrix and misclassification Error
pred <- predict(mymodel, testFL)

plot(pred) # D > B > A > C > E # Classes with highest wifi location

confusionMatrix(testFL$FL, pred) # Accuracy : 0.9686 and Kappa : 0.9593


# Misclassification Error
tab <- table(predicted= pred, Actual= testFL$FL)

1-sum(diag(tab))/sum(tab) # 0.03144865


# Accuracy
sum(diag(tab))/sum(tab) #0.9685514



# Building - Classification ---- # Random Forest

BUTrain<- as.data.frame(BUTrain[,])
BUTest <- as.data.frame(BUTest[,])

str(BUTrain)

#
RF_BU <- randomForest(BU~.,  BUTrain, metric = "Accuracy", importance = T,  trControl = ctrl2)


print(RF_BU)


# Prediction and Confusion Matrix -- Train 

Pred_RF <- predict(RF_BU, BUTest)

plot(Pred_RF) # Building 2 has the highest classified Wi-fi location

confusionMatrix(Pred_RF, BUTest$BU) # Accuracy : 0.9945 and Kappa : 0.9936 

# Misclassification Error
tab_BU <- table(Actual= BUTest$BU, predicted = Pred_RF) 

1-sum(diag(tab_BU))/sum(tab_BU) # 0.005

# Accuracy
sum(diag(tab_BU))/sum(tab_BU) # 0.0994


# Validation Dataset - RF

#
RF_BU1 <- randomForest(BU~.,
                      BU1Train, metric = "Accuracy", trControl = ctrlk)
print(RF_BU1)


# Prediction and Confusion Matrix -- Test 

Pred_RF_BU1 <- predict(RF_BU1, BU1Test)

plot(Pred_RF_BU1) # Building 0 has the highest classified variable

confusionMatrix(Pred_RF_BU1, BU1Test$BU) # Accuracy : 0.997 and Kappa : 0.9952

# Misclassification Error
tab_BU1 <- table(Actual= BU1Test$BU, predicted= Pred_RF_BU1)

table(Actual= BU1Train$BU, predicted= Pred_RF_BU1)

1-sum(diag(tab_BU1))/sum(tab_BU1) # 0.003

# Accuracy
sum(diag(tab_BU1))/sum(tab_BU1) # 0.0997





# Building - SVM 

BUTrain<- as.data.frame(BUTrain[,])
BUTest <- as.data.frame(BUTest[,])

mymodel_BU <- svm(BU~., BUTrain, kernel= 'radial', method = 'ctrlk')

print(mymodel_BU)

summary(mymodel_BU)


#Confusion Matrix and misclassification Error
pred_SVM_BU <- predict(mymodel_BU, BUTest)

plot(pred_SVM_BU) # Building 2 has the highest wifi and 1 is lowest

confusionMatrix(BUTest$BU, pred_SVM_BU) # Accuracy : 0.9975 and Kappa : 0.9961


# Misclassification Error
tab_Error_BU <- table(predicted= pred_SVM_BU, Actual= BUTest$BU)

1-sum(diag(tab_Error_BU))/sum(tab_Error_BU) # 0.0025


# Accuracy
sum(diag(tab_Error_BU))/sum(tab_Error_BU) #0.9975



# Building Validation DataSet - SVM

mymodel_BU1 <- svm(BU~., BU1Train, kernel= 'radial', method = 'ctrlk')

print(mymodel_BU1)

summary(mymodel_BU1)


#Confusion Matrix and misclassification Error
pred_SVM_BU1 <- predict(mymodel_BU1, BU1Test)

plot(pred_SVM_BU1) # Building 0 has the highest wifi and 2 is lowest

confusionMatrix(BU1Test$BU, pred_SVM_BU1) # Accuracy : 0.994 and Kappa : 0.9905


# Misclassification Error
tab_Error_BU1 <- table(predicted= pred_SVM_BU1, Actual= BU1Test$BU)

1-sum(diag(tab_Error_BU1))/sum(tab_Error_BU1) # 0.0030


# Accuracy
sum(diag(tab_Error_BU1))/sum(tab_Error_BU1) #0.9969



