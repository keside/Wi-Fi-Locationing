# --------------------------------------------------------------------------- #
# Ubiqum - Module 3 (IOT) - Wifi
# Author: Akano Keside
# Data exploration
# --------------------------------------------------------------------------- #

# Load Packages ----
library(tidyverse)
library(scatterplot3d)
library(ggplot2)
library(caret)
library(prcomp)
library(GGally)



# Load dataset ----
# Training Dataset - 19937 obs. of 529 Variable
# Validation Dataset - 1111obs. of 529 Variable
train<- read.csv('trainingData.csv', header = T, sep = ",")
validation<- read.csv('validationData.csv',header = T, sep = ",")


### Preliminary Analysis ----
head(train)
str(train)
glimpse(train)
any.omit(train)


### Data Processing ----

#-convert features to numeric
wifi_trainData <- sapply(train, as.numeric)  

wifi_validData <- sapply(validation, as.numeric) 

#-Convert matrix to tibble
wifi_trainData <- as_tibble(wifi_trainData)

wifi_validData <- as_tibble(wifi_validData)

glimpse(wifi_trainData)


#### I removed the variable i don't need in both dataset

wifi_trainData$TIMESTAMP <- NULL
wifi_trainData$USERID <- NULL


wifi_validData$TIMESTAMP <- NULL
wifi_validData$USERID <- NULL


# Created dependent and independent variable differently in both Dataset.

wap_train <- wifi_trainData[,1:520]

dep_trainData <- wifi_trainData[,521:526]


wap_valid <- wifi_validData[,1:520]

dep_validData <- wifi_validData[,521:526]


#-Convert features variable to categorical factor
dep_trainData$BUILDINGID <- factor(dep_trainData$BUILDINGID)
dep_trainData$SPACEID <- factor(dep_trainData$SPACEID)
dep_trainData$RELATIVEPOSITION <- factor(dep_trainData$RELATIVEPOSITION)
dep_trainData$FLOOR <- factor(dep_trainData$FLOOR)

dep_validData$BUILDINGID <- factor(dep_validData$BUILDINGID)
dep_validData$SPACEID <- factor(dep_validData$SPACEID)
dep_validData$RELATIVEPOSITION <- factor(dep_validData$RELATIVEPOSITION)
dep_validData$FLOOR <- factor(dep_validData$FLOOR)


#- I used Nearzerovar identify variable that have one unique value.

zeroVT <- nearZeroVar(wap_train,
                      freqCut = 1000,
                      uniqueCut = 0.1,
                     saveMetrics = FALSE)
head(zeroVT, 22)

#  Eliminated them from both datasets

new_waptrain <- wap_train[ , -zeroVT]
new_wapvalid <- wap_valid[ , -zeroVT]

dim(new_waptrain) # 19937   312
 

# Check if the datasets have same variable
all.equal(colnames(new_waptrain),
          colnames(new_wapvalid))


# Scaling.

normalize <- function(x){ return((x- min(x)) /(max(x)-min(x)))}

# normalize1 <- function(x) { return( (x + 104) / (0 + 104) ) }

train_norm <- as.data.frame(lapply(new_waptrain,
                                normalize))

valid_norm <- as.data.frame(lapply(new_wapvalid,
                                normalize))


## Principle Conponent Analysis : I used the principle for dimentionality reduction technique

princ <- prcomp(train_norm, scale = FALSE, center = FALSE)

#
head(princ,2)

#
princ$rotation[1:10, 1:9]

#
princ$sdev

#
dim(princ$x)


#compute standard deviation of each principal component
std_dev <- princ$sdev

#compute variance
pr_var <- std_dev^2

prop_var <- pr_var/sum(pr_var)

# Variance plot
plot(princ, xlab= 'var')



plot(prop_var, xlab= 'PC', ylab = 'Prop of variance', type = 'b')

#cumulative scree plot
plot(cumsum(prop_var), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b")


#Biplot 

biplot(princ, scale = 0)


#Scree Plot
plot(prop_var, type = 'line', main='scree plot')


# Rotation contains the loading which we are interested in
rotation <- princ$rotation


princ_T <- as.matrix(train_norm)

princ_V <- as.matrix(valid_norm)


brandnew_trainData <- princ_T %*% rotation
brandnew_validData <- princ_V %*% rotation



# I will also rename my independent variable for better typing.

colnames(dep_trainData) <-  c("LO", "LA", "FL", "BU", "SP", "RP")

colnames(dep_validData) <- c("LO", "LA", "FL", "BU", "SP", "RP")

comp <- 95


new_trainingSet <- as.data.frame(cbind(brandnew_trainData[,1:comp], dep_trainData))



new_validationSet <- as.data.frame(cbind(brandnew_validData[,1:comp], dep_validData))



# Saving my new dataset

write.csv(new_trainingSet, 'New_trainSet.csv', row.names = F)

write.csv(new_validationSet, 'New_validSet.csv', row.names = F)



# Data Visualization $ Exploring ----


# BuildingID

ggplot(new_trainingSet, aes(LO,
                         LA),
       colour = BUILDINGID) +
  ggtitle("Building ID - Vs Longitud $ Latitude") +
  geom_hex() +
  theme(legend.position = "bottom")



# Relative Position Plot

ggplot(new_trainingSet, aes(LO,
                         LA)) +
  geom_point(colour = new_trainSet$RP) +
  ggtitle("Relative Position")



# LO, LA and FL Plot

scatterplot3d(new_trainingSet$LO, new_trainingSet$LA, new_trainingSet$FL,
              type='p',
              highlight.3d = FALSE,
              color='blue',
              angle=155,
              pch=16,
              box=FALSE,
              main = "Location Reference Points Across Three Buildings",
              cex.lab = 1,
              cex.main=1,
              cex.sub=1,
              col.sub='blue',
              xlab='Longitude', ylab='Latitude',zlab = 'Building Floor')

# Plot - LA, LO, FL

scatterplot3d( x= new_trainingSet$LA,
               y= new_trainingSet$LO,
               z = new_trainingSet$FL,
               type = "p",
               color = new_trainingSet$SP,
               pch = 20)


# Correlation Plot

corr1 <- new_trainSet[,94:99]


ggcorr(corr1, label = TRUE)


ggpairs(corr1)