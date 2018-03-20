
library(rpart)
library(rpart.plot)
library(ModelMetrics)
library(caret)
source("0_split_data.R")
credit_model_gini <- rpart(formula = default ~ ., 
                       data = credit_train, 
                       method = "class",
                       parms = list(split = "gini"))

# Train an information-based model
credit_model_info<- rpart(formula = default ~ ., 
                       data = credit_train, 
                       method = "class",
                       parms = list(split = "information"))
credit_model_bag <- bagging(formula = default ~ .,  data = credit_train,    coob = TRUE)
credit_model_rf <- randomForest(formula = default ~ .,    data = credit_train)
