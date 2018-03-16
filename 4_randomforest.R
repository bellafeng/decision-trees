library(randomForest)
source("split_data.R")
library(ipred)

library(ModelMetrics)
library(caret)
# Train a Random Forest
set.seed(1)  # for reproducibility
credit_model <- randomForest(formula = default ~ ., 
                             data = credit_train)

# Print the model output                             
print(credit_model)

# Grab OOB error matrix & take a look
err <- credit_model$err.rate
head(err)

# Look at final OOB error rate (last row in err matrix)
oob_err <- err[nrow(err), "OOB"]
print(oob_err)

# Plot the model trained in the previous exercise
plot(credit_model)

# Add a legend since it doesn't have one by default
legend(x = "right", 
       legend = colnames(err),
       fill = 1:ncol(err))

# Generate predicted classes using the model object
class_prediction <- predict(object =credit_model ,   # model object 
                            newdata = credit_test,  # test dataset
                            type = "class") # return classification labels

# Calculate the confusion matrix for the test set
cm <- confusionMatrix(data = class_prediction,       # predicted classes
                      reference = credit_test$default)  # actual classes
print(cm)

# Compare test set accuracy to OOB accuracy
paste0("Test Accuracy: ", cm$overall[1])
paste0("OOB Accuracy: ", 1 - oob_err)