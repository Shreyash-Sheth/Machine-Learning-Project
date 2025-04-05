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
