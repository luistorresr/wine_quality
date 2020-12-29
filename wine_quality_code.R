###########################################################################################
###########################   Wine Quality  Project  ######################################
################################# Luis D. Torres ##########################################
###########################################################################################

# download key packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

options(digits = 3) # decimal points to 3

# Downloading the dataset

### https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
### https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

url_data <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

wine_quality_data <- read.csv2(url_data) # dataset

rm(url_data) # deleting temporary files 

# Inspecting wine_quality_data

class(wine_quality_data)
head(wine_quality_data)
lapply(wine_quality_data, class) # variables are in different data types although they are numeric 

# Transforming wine_quality_data variables into numeric

wine_quality_data[,1:12] <- lapply(wine_quality_data[,1:12], as.numeric)
lapply(wine_quality_data, class)

# summarising the wine_quality_data information 

if(!require(psych)) install.packages("psych", repos = "http://cran.us.r-project.org")

library(psych)

describe(wine_quality_data) %>% select(n, mean, sd, median, min, max) %>% knitr::kable("pipe", digits = 2)

# Partitioning the data intro training and validation. Validation set will be 20% of wine_quality_data

set.seed(1, sample.kind="Rounding")

index <- createDataPartition(y = wine_quality_data$quality, times = 1, p = 0.2, list = FALSE)
training <- wine_quality_data[-index,]
validation <- wine_quality_data[index,]
rm(index)# remove index

########################## Working with the training set ############################

# Partitioning the training set into train and test. Test is the 20% of the training dataset

set.seed(1, sample.kind="Rounding")

index <- createDataPartition(y = training$quality, times = 1, p = 0.2, list = FALSE)
train <- training[-index,]
test <- training[index,]
rm(index)# remove index

# Preprocessing 

if(!require(psych)) install.packages("psych", repos = "http://cran.us.r-project.org")
library(psych)

describe(train) %>% select(mean, sd, median, min, max) %>% knitr::kable("pipe", digits = 2)

## reviewing the outcome variable

library(ggplot2)
wine_quality_raw <-ggplot(train, aes(x=factor(quality))) + 
  geom_bar() + 
  xlab("Wine quality")

wine_quality_raw

## checking that the predictors have enough variability 

predictors <- train %>% select(fixed.acidity, volatile.acidity, citric.acid, residual.sugar,
                               chlorides, free.sulfur.dioxide, total.sulfur.dioxide, density, pH, 
                               sulphates, alcohol) %>% as.matrix()

nzv <- caret::nearZeroVar(predictors, saveMetrics= TRUE, names = TRUE)
nzv %>% select(zeroVar) %>% knitr::kable("rst", digits = 2) # no zero variation variables

# Data transformations

## Transforming the outcome variable

train <- train %>% mutate(levels = recode(quality, 
                                          "1" = "bad",
                                          "2" = "bad",
                                          "3" = "bad",
                                          "4" = "bad",
                                          "5" = "medium",
                                          "6" = "medium",
                                          "7" = "good",
                                          "8" = "good",
                                          "9" = "good",
                                          "10" = "good")) # for training

test <- test %>% mutate(levels = recode(quality, 
                                        "1" = "bad",
                                        "2" = "bad",
                                        "3" = "bad",
                                        "4" = "bad",
                                        "5" = "medium",
                                        "6" = "medium",
                                        "7" = "good",
                                        "8" = "good",
                                        "9" = "good",
                                        "10" = "good")) # same for test

validation <- validation %>% mutate(levels = recode(quality, 
                                                    "1" = "bad",
                                                    "2" = "bad",
                                                    "3" = "bad",
                                                    "4" = "bad",
                                                    "5" = "medium",
                                                    "6" = "medium",
                                                    "7" = "good",
                                                    "8" = "good",
                                                    "9" = "good",
                                                    "10" = "good")) # same for validation
## checking variables and frecuency of observations

library(ggplot2)
wine_quality_levels <- ggplot(train, aes(x=factor(levels, level = c('bad', 'medium', 'good')))) + 
  geom_bar() + 
  xlab("Wine quality levels")

wine_quality_levels

library(gridExtra)
grid.arrange(wine_quality_raw, wine_quality_levels, ncol=2) # comparing the distribution of observations 

# Fitting models using k-nearest neighbours, decision tree and random forests 

## Model 1 raw: k-nearest neighbours

set.seed(123, sample.kind="Rounding")

fit_knn <- train(factor(quality) ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar +
                   chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + 
                   sulphates + alcohol, method = "knn", 
                 data = train,
                 trControl = trainControl(method="cv", number = 5, p = .9),
                 tuneGrid = data.frame(k = seq(5, 100, 1)))

ggplot(fit_knn, highlight = TRUE)
fit_knn$bestTune #  parameter that maximized the accuracy
fit_knn$finalModel # best performing model
y_hat_knn <- predict(fit_knn, test, type = "raw")
acc_1 <- confusionMatrix(y_hat_knn, factor(test$quality))$overall["Accuracy"]
acc_1 # accuracy 

accuracy_models <- tibble(Method = "Model 1 raw: k-nearest neighbours", Accuracy = acc_1) # summary


## Model 2 raw: Decision trees

library(rpart)
### use cross validation to choose parameter
set.seed(300, sample.kind="Rounding")

fit_rtree <- train(factor(quality) ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar +
                     chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + 
                     sulphates + alcohol, data = train,
                   method = "rpart",
                   tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)))

ggplot(fit_rtree, highlight = TRUE)

fit_rtree$bestTune$cp # best cp value

plot(fit_rtree$finalModel, margin = 0.1)
text(fit_rtree$finalModel, cex = 0.75)

y_hat_rtree <- predict(fit_rtree, test)
acc_2 <- confusionMatrix(y_hat_rtree, factor(test$quality))$overall["Accuracy"]
acc_2 # accuracy

accuracy_models <- bind_rows(accuracy_models,
                             tibble(Method="Model 2 raw: Decision trees",
                                    Accuracy = acc_2)) # summary 

### retrieving the predictors in the tree
ind <- !(fit_rtree$finalModel$frame$var == "<leaf>") 
rtree_terms <- 
  fit_rtree$finalModel$frame$var[ind] %>%
  unique() %>%
  as.character()
rtree_terms

## Model 3 raw: Random forests

library(randomForest)
set.seed(999, sample.kind="Rounding")

### cross-validation to choose parameter

train_rforest <- train(factor(quality) ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar +
                         chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + 
                         sulphates + alcohol, 
                       method = "rf", data = train,
                       trControl = trainControl(method="cv", number = 5),
                       tuneGrid = data.frame(mtry = c(1:10))) # optimised algorithm

ggplot(train_rforest, highlight = TRUE)

train_rforest$bestTune

fit_rforest <- randomForest(factor(quality) ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar +
                              chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + 
                              sulphates + alcohol, 
                            data = train, minNode = train_rforest$bestTune$mtry) # fit optimised model

y_hat_rforest <- predict(fit_rforest, test)
acc_3 <- confusionMatrix(y_hat_rforest, factor(test$quality))$overall["Accuracy"]
acc_3 # # accuracy

accuracy_models <- bind_rows(accuracy_models,
                             tibble(Method="Model 3 raw: Random forest",
                                    Accuracy = acc_3)) # summary 

fit_rforest$importance %>% knitr::kable("rst", digits = 2) # importance of each feature


## Model 1b levels: k-nearest neighbours

set.seed(123, sample.kind="Rounding")

fit_knn_b <- train(levels ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar +
                 chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + 
                 sulphates + alcohol, method = "knn", 
                   data = train,
                   trControl = trainControl(method="cv", number = 5, p = .9),
                   tuneGrid = data.frame(k = seq(5, 100, 1)))

ggplot(fit_knn_b, highlight = TRUE)
fit_knn_b$bestTune #  parameter that maximised the accuracy
fit_knn_b$finalModel # best performing model
y_hat_knn_b <- predict(fit_knn_b, test, type = "raw")
acc_4 <- confusionMatrix(y_hat_knn_b, factor(test$levels))$overall["Accuracy"]
acc_4 # accuracy

accuracy_models <- bind_rows(accuracy_models,
                             tibble(Method="Model 1 levels: k-nearest neighbours",
                                    Accuracy = acc_4)) # summary

## Model 2b levels: Decision trees

library(rpart)
### use cross validation to choose parameter
set.seed(300, sample.kind="Rounding")

fit_rtree_b <- train(levels ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar +
                     chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + 
                     sulphates + alcohol, data = train,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)))

ggplot(fit_rtree_b, highlight = TRUE)

fit_rtree_b$bestTune$cp # best cp

plot(fit_rtree_b$finalModel, margin = 0.1)
text(fit_rtree_b$finalModel, cex = 0.75)
y_hat_rtree_b <- predict(fit_rtree_b, test)
acc_5 <- confusionMatrix(y_hat_rtree_b, factor(test$levels))$overall["Accuracy"]
acc_5 # accuracy
accuracy_models <- bind_rows(accuracy_models,
                             tibble(Method="Model 2 levels: Decision trees",
                                    Accuracy = acc_5)) # summary

### Retrieving the predictors in the tree

ind_b <- !(fit_rtree_b$finalModel$frame$var == "<leaf>") 
rtree_terms_b <- 
  fit_rtree_b$finalModel$frame$var[ind_b] %>%
  unique() %>%
  as.character()
rtree_terms_b


## Model 3b levels: Random forests
library(randomForest)
set.seed(999, sample.kind="Rounding")

### cross validation to choose parameter

train_rforest_b <- train(levels ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar +
                         chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + 
                         sulphates + alcohol, 
                   method = "rf", data = train,
                   trControl = trainControl(method="cv", number = 5),
                   tuneGrid = data.frame(mtry = c(1:10))) # optimised algorithm
ggplot(train_rforest_b, highlight = TRUE)
train_rforest_b$bestTune

fit_rforest_b <- randomForest(factor(levels) ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar +
                              chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + 
                              sulphates + alcohol, data = train, minNode = train_rforest$bestTune$mtry) # fit optimised model

y_hat_rforest_b <- predict(fit_rforest_b, test)
acc_6 <- confusionMatrix(y_hat_rforest_b, factor(test$levels))$overall["Accuracy"]
acc_6  # accuracy
accuracy_models <- bind_rows(accuracy_models,
                             tibble(Method="Model 3 levels: Random forest",
                                    Accuracy = acc_6)) # summary

fit_rforest_b$importance %>% knitr::kable("rst", digits = 2) # importance of each feature

# Validation with the best performing model

y_hat_rforest_validation <- predict(fit_rforest_b, validation)
acc_validation <- confusionMatrix(y_hat_rforest_validation, factor(validation$levels))$overall["Accuracy"]
acc_validation # # accuracy

accuracy_models <- bind_rows(accuracy_models,
                             tibble(Method="Best performing model validation",
                                    Accuracy = acc_validation)) # summary

# Model comparison 

accuracy_models %>% knitr::kable()



