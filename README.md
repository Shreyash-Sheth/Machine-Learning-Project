# Dataset
I have chosen the Titanic dataset for this project, which provides details about one of the most notorious shipwrecks in history. This dataset includes passenger information, including age, gender, ticket fare, and the booked cabin class. Additionally, it provides crucial information about whether each passenger survived the tragic shipwreck.

# Exploring the Data
Exploring the data through various functions available within the ‘Tidyverse’ package to gain insights into the dataset and datatype.

       
       # Using the tidyverse library to explore the dataset
       install.packages("tidyverse")
       library(tidyverse)
       
       d <- read.csv("C:\\Users\\yashs\\Desktop\\BA\\IS 6052 Predictive Analytics\\Assignment\\Titanic.csv")

       # exploring dimensions of the dataset
       dim(d)
       # exploring structure of the dataset
       str(d)
       # glimpse of the dataset
       glimpse(d)

       head(d)
       tail(d)

       attach(d)

       view(sort(table(Gender), decreasing = TRUE))
       view(sort(table(Cabin.Class), decreasing = TRUE))
       view(sort(table(Survived), decreasing = TRUE))
       view(sort(table(Age), decreasing = TRUE))

       # viewing the number of missing values in the column
       view(d[is.na(Age), ])
       
       boxplot(Age)
  ![image](https://github.com/user-attachments/assets/3f778411-e809-41df-86bd-0efabccbd26b)

The dataset has 891 rows (observations) and 5 columns (variables). Columns available are Survived (Integer), Cabin Class (Integer), Gender (Character), Age (Numerical) and Fare (Numerical).

# Data Preparation
Before delving into the dataset, my initial step involves preparing the data for analysis. I begin by observing the dataset to identify and address any missing values. Upon observation, I noticed that the 'Fare' column contains values of '0'. Given the small size of the dataset and assuming that some passengers may have boarded the ship for free, I decided to leave these values unchanged.
 
 ![image](https://github.com/user-attachments/assets/05d6c608-f33a-4ed9-8ae9-dc649c696c81)
 
The entries in the "survived" column have been transformed from '1' and '0' to 'Survived' and 'Died' for better comprehension. Using the 'unique' function, it becomes apparent that the ‘Age’ column contains numerous missing values (NA). An examination of the 'Age' column through plotting reveals a right-skewed distribution, leading to the replacement of missing values with the median age. Additionally, considering the presence of outliers in the 'Age' column, I find it more appropriate to replace the missing values with the median value. This approach is preferable as the median is not affected by outliers compared to other measures. Moreover, the "Survived," "Gender," and "Cabin Class" columns were converted to factors for ease of analysis.

  ![image](https://github.com/user-attachments/assets/8a80a845-4a12-45c6-879e-9f2520a94669)

  ![image](https://github.com/user-attachments/assets/cb6aa304-be1c-463b-8adc-487a15be8e6a)

  ![image](https://github.com/user-attachments/assets/fdf92888-22c2-44e6-a270-6199d09eb950)

  ![image](https://github.com/user-attachments/assets/64327d69-b43a-40cf-88f1-b97d9527b5fd)

The median age ‘28’ is used to replace all the missing values in the Age column.
The new dataset has been saved to the variable ‘d2’. I will use this new dataset for all further analysis.  

# Exploratory Data Analysis
Using exploratory data analysis to examine the correlation between each variable and the 'Survived' column. This helps in comprehending the impact of different variables on the survival rate, which will be beneficial when choosing a feature for splitting in tree-based models.

Plotting the overall survival rate:
![image](https://github.com/user-attachments/assets/e917700b-7b93-4ddf-8666-30d16bf7fd34)

Passenger survival rate:
![image](https://github.com/user-attachments/assets/8cac923a-6794-4590-b02d-43144201296e)
The number of passengers that died are more than the ones that survived. Now, let’s visualize the survival rates based on gender and cabin class.

Plotting the effect of gender on survival:
![image](https://github.com/user-attachments/assets/c4983b1e-ad11-43cc-b331-40c0606a2021)

Effect of gender on survival:
![image](https://github.com/user-attachments/assets/162de9b3-aab8-4533-afe5-69c844e0d942)
From the visual representation, I can conclude that approximately 70% of females survived, whereas only around 30% of males survived the shipwreck. Considering gender as a variable in our predictive analysis could prove to be significant.

Using the same code as above but replacing the Gender column with Cabin Class, I can get the below graph. The survival rate for passengers in third class was notably low, around 30%, in contrast to a higher survival rate of approximately 70% for passengers in first class.
![image](https://github.com/user-attachments/assets/07d395bb-7414-4641-b1bd-55bca670f9da)

# Predictive Analysis
The predictive analysis will be conducted on the 'd2' dataset, employing three distinct predictive models: Decision Tree, Random Forest, and k-NN. The objective is to predict the accuracy of the models in determining whether passengers will survive or die in the shipwreck. In the absence of dedicated test data, the existing dataset is divided into training and testing data. Specifically, 75% of the observations will be used for training (train), and the remaining 25% for testing (test). The ‘CreateDataPartition’ function from the ‘caret’ package will be utilized for this purpose.

R script for data partitioning:
![image](https://github.com/user-attachments/assets/633a2f28-501b-4ccb-bec1-43735233906e)

The dataset is created as seen on the environment window:
![image](https://github.com/user-attachments/assets/a45a8ea1-5a0c-4d3f-b535-4c88e66c2842)

## Decision Tree
The decision tree model starts with all observations at the root node and selects the most appropriate variable to split the data from all available variables; this is based on the concept of information gain. After the initial split, the process iterates by moving down one level, making subsequent splits. The nodes situated at the bottom of the decision tree are referred to as terminal nodes. The prediction for new observations that reach a terminal node is determined by the majority vote of the observations within that node. For constructing the decision tree model, the rpart package is employed. The model is trained using the training data, and the decision tree is visualized using the rpart.plot function. All variables are considered in the model.

       # Set control parameters
       control <- rpart.control(minsplit = 1, minbucket = 1 maxdepth = 10, cp = 0.01)
       # Train the model
       decisionT <- rpart(train$Survived~., method = 'class', data = train, control = control) 
       # Plotting the model
       rpart.plot(decisionT, extra=108)

Decision Tree Plot:
![image](https://github.com/user-attachments/assets/44412885-5702-49a0-b295-f82fa09afc29)

After establishing the model, the testing data is utilized for the prediction and assessment of the model's accuracy. The ‘predict’ function is applied to make predictions using the test data, with the type parameter set to "class" for classification purposes. Subsequently, a confusion matrix is generated to gauge the accuracy of the model.

       # Prediction on the test data
       predictions <- predict(decisionT,test, type = 'class')
       # Confusion Matrix
       confusionMatrix(predictions,test$Survived)

Console output of confusionMatrix:
![image](https://github.com/user-attachments/assets/7abd24d2-fe85-437c-90f8-e5d828136874)

The model correctly predicts 120 instances of ‘Died’ and 63 instances of ‘Survived’ but misclassifies 39 instances. The model gives us an accuracy of 82.43%

## Random Forest
Random forest is a supervised learning model that combines prediction of multiple decision trees. Each tree is trained on a random subset of data considering random subset of features at each split. I will use the same ‘train’ and ‘test’ data for this model as well.

R script for training the random forest model:

       library(randomForest)
       # Train the model
       randomF <- randomForest(train$Survived~., data = train, ntree = 500, mtry = 2)
       # Display model details
       print(randomF)

Random Forest model console output:
![image](https://github.com/user-attachments/assets/023ab417-4dd8-4c6c-bdf1-29c78496dba5)

The out-of-bag (OOB) estimate error rate, standing at 17.64%, provides an approximation of the model's accuracy on unseen data. This suggests an expected error rate of around 17.64%. The class error metric offers insights into the accuracy of predictions for each class.
Having gained a comprehensive understanding of the model's accuracy, although not mandatory, I will proceed to assess its performance further. Applying the testing data to the model will allow us to evaluate its efficacy, and subsequently, I will derive the confusion matrix for a more detailed examination.

       # Prediction on the test data
       predictRF <- predict(randomF,test, type = 'class')
       # Confusion Matrix
       confusionMatrix(predictRF,test$Survived)

Console output of confusionMatrix:
![image](https://github.com/user-attachments/assets/9ee7c4fa-4028-450c-b45d-4b14de5a4875)

The model correctly predicts 122 instances of ‘Died’ and 60 instances of ‘Survived’ but misclassifies 45 instances. The model gives us an accuracy of 81.98%

## k-NN
k-Nearest Neighbors (k-NN) is a supervised learning algorithm primarily employed for classification tasks. It classifies a data point based on the classification of its nearest neighbors. To ensure reproducibility, I established a seed of '12345'. For training purposes, control parameters are configured using a 10-fold Cross-Validation method. During model training, the 'tuneLength' is set to 20, denoting the number of values to explore for the tuning parameter (k). This signifies an evaluation of a range of values for k to identify the optimal one.

R script for training the k-NN model:

       # Set seed for reproducibility
       set.seed(12345)
       # Set control parameters
       ctrl <- trainControl(method = 'cv', number = 10)
       # Train the model
       knn <- train(Survived~., data = train, method = 'knn', trControl = ctrl, tuneLength = 20)
       # Display model details
       print(knn)

Output of the k-NN model:
![image](https://github.com/user-attachments/assets/617c601d-410c-48dd-854c-6c569a07b85b)

R script for Prediction and confusionMatrix:

       # Plot the model
       plot(knn)
       # Prediction on the test data
       knnPredict <- predict(knn, newdata=test)
       # Confusion Matrix
       confusionMatrix(knnPredict, test$Survived)
       print(conf_matrix)

k-NN plot:
![image](https://github.com/user-attachments/assets/953e727f-2258-4465-a9b4-9fc73d52b43a)

Console output of confusionMatrix:
![image](https://github.com/user-attachments/assets/885709c3-11f8-44b5-8db0-5e0f7caec083)

The optimal value for k in the model is identified as 5, where it yields the highest accuracy. Utilizing the 'predict' function, the model makes predictions on the test data, and the 'confusionMatrix' is employed to assess its performance.
The model accurately predicts 108 instances of 'Died' and 47 instances of 'Survived,' but it misclassifies 67 instances. Consequently, the model achieves an accuracy rate of 69.82%.

# Conclusion

After the analysis, the accuracy for each model is summarized below:

- Decision Tree - 82.43%
- Random Forest - 81.98%
- k-NN - 69.82%

The decision tree gives us the best accuracy in predicting the survival rate on test data, followed by Random Forest and k-NN.

