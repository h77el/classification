# load needed libraries
library(DBI)
library(tidyverse)
library(lubridate)
library(visdat)
library(randomForest)

# get dataframes
df <- read.csv(url("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv"))
df$Personal.Loan <- factor(df$Personal.Loan)


# Data preprocessing
# Check for missing values in each column
vis_miss(df)

# Baseline model
# Split the dataset into training and testing sets
set.seed(1) # Set random seed for reproducibility
train_index <- sample(1:nrow(df), 0.7 * nrow(df)) # 30% of the data as training set
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Build random forest model
rf_model <- randomForest(Personal.Loan ~ ., data = train_data)

# Make predictions on the test set
predictions <- predict(rf_model, newdata = test_data)


# Define function
compute_classification_metrics <- function(actual_labels, predicted_labels) {
  # Compute confusion matrix
  cm <- table(actual_labels, predicted_labels)
  
  # Calculate true positives (TP), false negatives (FN), false positives (FP), and true negatives (TN)
  TN <- cm[1, 1]
  FN <- cm[2, 1]
  FP <- cm[1, 2]
  TP <- cm[2, 2]
  
  # Calculate accuracy
  accuracy <- (TP + TN) / sum(cm)
  
  # Calculate recall
  recall <- TP / (TP + FN)
  
  # Calculate precision
  precision <- TP / (TP + FP)
  
  # Calculate F1 score
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Store the results in a list
  metrics <- list(
    confusion_matrix = cm,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score
  )
  
  return(metrics)
}

# Model evaluation
metrics <- compute_classification_metrics(test_data$Personal.Loan, predictions)

# Print the results
print(metrics)


# Improvement method
# Check
# Calculate the number of unique values in each column
unique_counts <- sapply(df, function(col) length(unique(col)))

# Create a data frame to store the number of unique values and column names
unique_counts_df <- data.frame(column = names(unique_counts), unique_count = unique_counts)


# Visualize the number of unique values using ggplot2
ggplot(unique_counts_df, aes(x = column, y = unique_count)) +
  geom_bar(stat = "identity") +
  labs(x = "Column", y = "Unique Value Count") +
  ggtitle("Number of Unique Values in Each Column")


# Convert categorical variables to dummy variables
# Select column names with fewer than 10 unique values and exclude "Personal.Loan"
selected_columns <- unique_counts_df$column[unique_counts_df$unique_count < 10 & unique_counts_df$column != "Personal.Loan"]

# Print column names with fewer than 10 unique values (excluding "Personal.Loan")
print(selected_columns)

# Column names to convert
columns_to_convert <- selected_columns
df$Personal.Loan <- as.integer(df$Personal.Loan) - 1

# Loop through the columns to convert
for (col in columns_to_convert) {
  # Convert the column to a factor variable
  df[[col]] <- factor(df[[col]])
}


# Convert variables in the data frame to dummy variables
dummy_variables <- model.matrix(~ . - 1, df)

# Store the converted dummy variables in a new data frame
data <- as.data.frame(dummy_variables)
data$Personal.Loan <- factor(data$Personal.Loan)

# Count the number of each label category
label_counts <- table(data$Personal.Loan)

# Calculate percentages
percentages <- round(100 * label_counts / sum(label_counts), 1)

# Plot a pie chart showing the label distribution
pie(label_counts,
    labels = paste0(c("0: ", "1: "), percentages, "%"),
    col = c("lightgreen", "lightblue"),
    main = "Label Distribution")


# Data augmentation
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Expand positive samples in the training set
positive_samples <- train_data[train_data$Personal.Loan == 1, ]
negative_samples <- train_data[train_data$Personal.Loan == 0, ]
positive_samples_expanded <- positive_samples[sample(1:nrow(positive_samples), nrow(negative_samples), replace = TRUE), ]
train_data_balanced <- rbind(positive_samples_expanded, negative_samples)

# Build a random forest model
rf_model <- randomForest(Personal.Loan ~ ., data = train_data_balanced)

# Make predictions on the test set
predictions2 <- predict(rf_model, newdata = test_data)

# Model evaluation
metrics2 <- compute_classification_metrics(test_data$Personal.Loan, predictions2)

# Print the results
print(metrics2)


# Hyperparameter selection
# Set different numbers of trees
ntrees <- c(10, 50, 100, 200, 300, 500, 1000)

# Initialize a vector to store recall at different numbers of trees
recalls <- numeric(length(ntrees))

set.seed(1)

# Train random forest models and calculate recall
for (i in seq_along(ntrees)) {
  
  # Build the random forest model
  rf_model <- randomForest(Personal.Loan ~ ., 
                           data = train_data_balanced, 
                           ntree = ntrees[i])
  
  # Predict on the test set
  predictions <- predict(rf_model, newdata = test_data)
  
  # Calculate recall
  metrics <- compute_classification_metrics(test_data$Personal.Loan, predictions)
  recalls[i] <- metrics$recall
}

# Plot recall curve
plot(ntrees, recalls, 
     type = "b", 
     xlab = "Number of Trees", ylab = "Recall", 
     main = "Recall with Number of Trees")

# Set x-axis ticks to 50
axis(1, at = seq(0, max(ntrees), by = 50))

# Final model
rf_model <- randomForest(Personal.Loan ~ ., 
                         data = train_data_balanced, 
                         ntree = 300)

# Predict on the test set
predictions <- predict(rf_model, newdata = test_data)

# Calculate recall
metrics <- compute_classification_metrics(test_data$Personal.Loan, predictions)
print(metrics)

# Calculate variable importance
var_importance <- importance(rf_model)
print(var_importance)

# Visualize variable importance
varImpPlot(rf_model)


