library(rpart)
library(caret)

# Set control parameters
control <- rpart.control(minsplit = 1, minbucket = 1, maxdepth = 10, 
                         cp = 0.01)
# Train the model
decisionT <- rpart(train$Survived~., method = 'class', data = train, 
                   control = control)
# Plotting the model
rpart.plot(decisionT, extra=108)
# Prediction on the test data
predictions <- predict(decisionT,test, type = 'class')
# Confusion Matrix
confusionMatrix(predictions,test$Survived)

library(randomForest)
# Train the model
randomF <- randomForest(train$Survived~., data = train, ntree = 500, mtry = 2)
# Display model details
print(randomF)
# Prediction on the test data
predictRF <- predict(randomF,test, type = 'class')
# Confusion Matrix
confusionMatrix(predictRF,test$Survived)

# Set seed for reproducibility
set.seed(12345)
# Set control parameters
ctrl <- trainControl(method = 'cv', number = 10)
# Train the model
knn <- train(Survived~., data = train, method = 'knn', trControl = ctrl, 
             tuneLength = 20)
# Display model details
print(knn)
# Plot the model
plot(knn)
# Prediction on the test data
knnPredict <- predict(knn, newdata=test)
# Confusion Matrix
confusionMatrix(knnPredict, test$Survived)
print(conf_matrix)
# Visualizing the confusion matrix
ggplot(data = as.data.frame(conf_matrix$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix for k-NN Model",
       x = "Reference",
       y = "Prediction") +
  theme_minimal()
