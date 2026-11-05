
# ML Model Comparison & Additional Evaluation

# Install and load packages
library(class)
library(e1071)
library(rpart)
library(randomForest)

# Load dataset
students <- read.csv("StudentsPerformance.csv")

# Create average score
students$average_score <- (
  students$math.score +
    students$reading.score +
    students$writing.score
) / 3

# Create target variable
students$result <- factor(
  ifelse(students$average_score >= 60, "Pass", "Needs Improvement"),
  levels = c("Needs Improvement", "Pass")
)

# Select input features
data <- students[, c(
  "gender",
  "race.ethnicity",
  "parental.level.of.education",
  "lunch",
  "test.preparation.course",
  "result"
)]

# Split dataset into training and testing data
set.seed(123)

train_index <- sample(
  1:nrow(data),
  size = 0.8 * nrow(data)
)

train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Separate target variable
y <- data$result
y_train <- train_data$result
y_test <- test_data$result

# Convert categorical features into numeric format
x_train <- model.matrix(result ~ ., data = train_data)[, -1]
x_test <- model.matrix(result ~ ., data = test_data)[, -1]

# Scale data for KNN and SVM
x_train_scaled <- scale(x_train)

x_test_scaled <- scale(
  x_test,
  center = attr(x_train_scaled, "scaled:center"),
  scale = attr(x_train_scaled, "scaled:scale")
)

# Store predictions
predictions <- list()

# 1. KNN Model
predictions[["KNN"]] <- knn(
  train = x_train_scaled,
  test = x_test_scaled,
  cl = y_train,
  k = 3
)

# 2. SVM Model
svm_model <- svm(
  x = x_train_scaled,
  y = y_train,
  kernel = "radial",
  cost = 1,
  gamma = 1
)

predictions[["SVM"]] <- predict(
  svm_model,
  x_test_scaled
)

# 3. Naive Bayes Model
nb_model <- naiveBayes(
  result ~ .,
  data = train_data
)

predictions[["Naive Bayes"]] <- predict(
  nb_model,
  test_data
)

# 4. Decision Tree Model
tree_model <- rpart(
  result ~ .,
  data = train_data,
  method = "class"
)

predictions[["Decision Tree"]] <- predict(
  tree_model,
  test_data,
  type = "class"
)

# 5. Random Forest Model
rf_model <- randomForest(
  result ~ .,
  data = train_data,
  ntree = 500
)

predictions[["Random Forest"]] <- predict(
  rf_model,
  test_data
)

# Create folder for confusion matrices
dir.create("confusion_matrices", showWarnings = FALSE)

# Store model results
results <- data.frame()

# Store class-wise performance
detailed_results <- data.frame()

# Evaluate every model
for (model_name in names(predictions)) {
  
  pred <- factor(
    predictions[[model_name]],
    levels = levels(y)
  )
  
  # Confusion matrix
  cm <- table(
    Actual = factor(y_test, levels = levels(y)),
    Predicted = pred
  )
  
  # Save confusion matrix
  write.csv(
    as.data.frame.matrix(cm),
    paste0("confusion_matrices/", model_name, ".csv")
  )
  
  # Calculate accuracy
  accuracy <- sum(diag(cm)) / sum(cm) * 100
  
  # Store class-wise metrics
  for (cl in levels(y)) {
    
    TP <- cm[cl, cl]
    FP <- sum(cm[, cl]) - TP
    FN <- sum(cm[cl, ]) - TP
    
    precision <- ifelse(
      TP + FP == 0, 0, TP / (TP + FP)
    )
    
    recall <- ifelse(
      TP + FN == 0, 0, TP / (TP + FN)
    )
    
    f1 <- ifelse(
      precision + recall == 0,
      0,
      2 * precision * recall / (precision + recall)
    )
    
    detailed_results <- rbind(
      detailed_results,
      data.frame(
        Model = model_name,
        Class = cl,
        Precision = precision * 100,
        Recall = recall * 100,
        F1_Score = f1 * 100
      )
    )
  }
  
  # Calculate macro F1 Score
  model_f1 <- mean(
    detailed_results$F1_Score[
      detailed_results$Model == model_name
    ]
  )
  
  # Store overall results
  results <- rbind(
    results,
    data.frame(
      Model = model_name,
      Accuracy = accuracy,
      F1_Score = model_f1
    )
  )
}

# Save model comparison results
write.csv(
  results,
  "model_comparison_results.csv",
  row.names = FALSE
)

# Save detailed class-wise performance
write.csv(
  detailed_results,
  "model_classwise_performance.csv",
  row.names = FALSE
)

# Save model predictions
prediction_results <- data.frame(
  Actual = y_test
)

for (model_name in names(predictions)) {
  prediction_results[[model_name]] <- predictions[[model_name]]
}

write.csv(
  prediction_results,
  "model_predictions.csv",
  row.names = FALSE
)

# Create accuracy comparison plot
png(
  "model_comparison_accuracy.png",
  width = 1000,
  height = 700
)

barplot(
  results$Accuracy,
  names.arg = results$Model,
  main = "ML Model Accuracy Comparison",
  xlab = "Machine Learning Models",
  ylab = "Accuracy (%)",
  ylim = c(0, 100),
  las = 2
)

dev.off()

# Create F1 Score comparison plot
png(
  "model_comparison_f1_score.png",
  width = 1000,
  height = 700
)

barplot(
  results$F1_Score,
  names.arg = results$Model,
  main = "ML Model F1 Score Comparison",
  xlab = "Machine Learning Models",
  ylab = "Macro F1 Score (%)",
  ylim = c(0, 100),
  las = 2
)

dev.off()

# Find best model using Macro F1 Score
best_row <- results[
  which.max(results$F1_Score),
]

# Save best model report
write.csv(
  best_row,
  "best_model_report.csv",
  row.names = FALSE
)

# Display results
print(results)
print(detailed_results)
print(best_row)

cat("\nMini Project Completed Successfully!")
cat("\nAll 5 models evaluated successfully!")
cat("\nAll evaluation files and plots saved!")
