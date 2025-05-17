# ---------------- Load Libraries ----------------
library(caret)
library(pls)
library(randomForest)
library(e1071)
library(signal)      # For Savitzky-Golay filter
library(xgboost)
library(glmnet)
library(gbm)
library(ggplot2)
library(caretEnsemble)
library(GA)

# ---------------- Load Dataset ----------------
data1 <- read.csv(file.choose(), header = TRUE)
data <- data1[, -c(3, 2, 4, 5, 6, 7)]
data <- na.omit(data)

# ---------------- Preprocessing ----------------
nzv <- nearZeroVar(data)
if (length(nzv) > 0) {
  data <- data[, -nzv]
}

r2_score <- function(actual, predicted) {
  SS_res <- sum((actual - predicted)^2)
  SS_tot <- sum((actual - mean(actual))^2)
  return(ifelse(SS_tot == 0, NA, 1 - SS_res / SS_tot))
}

rmpe <- function(actual, predicted) {
  eps <- .Machine$double.eps
  return(mean(abs((actual - predicted) / (actual + eps))) * 100)
}

# ---------------- Preprocessing Functions ----------------
apply_sgd <- function(data_matrix) {
  sg_filtered <- apply(data_matrix, 2, function(col) {
    sgolayfilt(col, p = 2, n = 11)
  })
  return(as.data.frame(sg_filtered))
}

apply_snv <- function(data_matrix) {
  snv_transformed <- t(apply(data_matrix, 1, function(row) {
    row_sd <- sd(row)
    if (row_sd == 0) return(rep(0, length(row)))
    (row - mean(row)) / row_sd
  }))
  return(as.data.frame(snv_transformed))
}

apply_msc <- function(data_matrix) {
  ref <- colMeans(data_matrix)
  msc_data <- t(apply(data_matrix, 1, function(row) {
    fit <- lm(row ~ ref)
    if (length(coef(fit)) < 2 || coef(fit)[2] == 0) return(rep(0, length(row)))
    (row - coef(fit)[1]) / coef(fit)[2]
  }))
  return(as.data.frame(msc_data))
}

apply_minmax <- function(data_matrix) {
  return(as.data.frame(scale(data_matrix, center = apply(data_matrix, 2, min), scale = apply(data_matrix, 2, max) - apply(data_matrix, 2, min))))
}

apply_zscore <- function(data_matrix) {
  return(as.data.frame(scale(data_matrix)))
}

# ---------------- Choose Preprocessing Set ----------------
predictors <- data[, -1]
response <- data$N

preprocessing_methods <- list(
  SGD = apply_sgd,
  SNV = apply_snv,
  MSC = apply_msc,
  SNV_MSC = function(x) apply_msc(apply_snv(x)),
  SGD_MSC = function(x) apply_msc(apply_sgd(x)),
  SGD_SNV = function(x) apply_snv(apply_sgd(x)),
  SGD_SNV_MSC = function(x) apply_msc(apply_snv(apply_sgd(x)))
)

evaluate_model <- function(true, pred, model_name, dataset_type, preprocessing) {
  rmse <- sqrt(mean((true - pred)^2))
  r2 <- r2_score(true, pred)
  rmpe_val <- rmpe(true, pred)
  return(data.frame(Preprocessing = preprocessing, Model = model_name, Dataset = dataset_type, RMSE = rmse, R2 = r2, RMPE = rmpe_val))
}

all_results <- data.frame()
all_train_preds <- list()
all_test_preds <- list()

# ---------------- Loop Over Preprocessing Pipelines ----------------
for (method_name in names(preprocessing_methods)) {
  cat("\n\nRunning for preprocessing:", method_name, "\n")
  processed_data <- preprocessing_methods[[method_name]](predictors)
  data_processed <- cbind(N = response, processed_data)
  
  set.seed(123)
  trainIndex <- createDataPartition(data_processed$N, p = 0.8, list = FALSE)
  trainData <- data_processed[trainIndex, ]
  testData <- data_processed[-trainIndex, ]
  
  tryCatch({
    # ---------- PLSR ----------
    pls_model <- plsr(N ~ ., data = trainData, scale = TRUE, validation = "CV")
    ncomp_opt <- which.min(pls_model$validation$PRESS)
    pls_train_pred <- as.vector(predict(pls_model, trainData[, -1], ncomp = ncomp_opt)[,,1])
    pls_test_pred <- as.vector(predict(pls_model, testData[, -1], ncomp = ncomp_opt)[,,1])
    
    # ---------- Random Forest ----------
    rf_model <- randomForest(N ~ ., data = trainData, ntree = 500)
    rf_train_pred <- predict(rf_model, trainData[, -1])
    rf_test_pred <- predict(rf_model, testData[, -1])
    
    # ---------- SVM ----------
    svm_model <- train(N ~ ., data = trainData, method = "svmRadial",
                       trControl = trainControl(method = "cv", number = 5),
                       tuneLength = 5)
    svm_train_pred <- predict(svm_model, trainData[, -1])
    svm_test_pred <- predict(svm_model, testData[, -1])
    
    # ---------- XGBoost ----------
    xgb_model <- train(N ~ ., data = trainData, method = "xgbTree",
                       trControl = trainControl(method = "cv", number = 5),
                       tuneLength = 5)
    xgb_train_pred <- predict(xgb_model, trainData)
    xgb_test_pred <- predict(xgb_model, testData)
    
    # ---------- Elastic Net ----------
    enet_model <- train(N ~ ., data = trainData, method = "glmnet",
                        trControl = trainControl(method = "cv", number = 5),
                        tuneLength = 10)
    enet_train_pred <- predict(enet_model, trainData[, -1])
    enet_test_pred <- predict(enet_model, testData[, -1])
    
    # ---------- GBM ----------
    gbm_model <- train(N ~ ., data = trainData, method = "gbm",
                       distribution = "gaussian",
                       trControl = trainControl(method = "cv", number = 5),
                       tuneLength = 5, verbose = FALSE)
    gbm_train_pred <- predict(gbm_model, newdata = trainData)
    gbm_test_pred <- predict(gbm_model, newdata = testData)
    
    # ---------- Stacked Model ----------
    models_list <- tryCatch({
      caretList(N ~ ., data = trainData,
                trControl = trainControl(method = "cv", number = 5, savePredictions = "final"),
                methodList = c("rf", "svmRadial", "glmnet"))
    }, error = function(e) NULL)
    
    stacked_model <- tryCatch({
      if (!is.null(models_list)) caretStack(models_list, method = "glm") else NULL
    }, error = function(e) NULL)
    
    stacked_train_pred <- if (!is.null(stacked_model)) predict(stacked_model, newdata = trainData) else rep(NA, nrow(trainData))
    stacked_test_pred <- if (!is.null(stacked_model)) predict(stacked_model, newdata = testData) else rep(NA, nrow(testData))
    
    # ---------- PLS + SVM ----------
    pls_scores_train <- scores(pls_model)[, 1:ncomp_opt]
    pls_scores_test <- predict(pls_model, testData[, -1], type = "scores")[, 1:ncomp_opt]
    pls_svm_model <- train(x = pls_scores_train, y = trainData$N, method = "svmRadial",
                           trControl = trainControl(method = "cv", number = 5),
                           tuneLength = 5)
    pls_svm_train_pred <- predict(pls_svm_model, pls_scores_train)
    pls_svm_test_pred <- predict(pls_svm_model, pls_scores_test)
    
    # ---------- Combine Results ----------
    all_results <- rbind(
      all_results,
      evaluate_model(trainData$N, pls_train_pred, "PLSR", "Train", method_name),
      evaluate_model(testData$N, pls_test_pred, "PLSR", "Test", method_name),
      
      evaluate_model(trainData$N, rf_train_pred, "Random Forest", "Train", method_name),
      evaluate_model(testData$N, rf_test_pred, "Random Forest", "Test", method_name),
      
      evaluate_model(trainData$N, svm_train_pred, "SVM", "Train", method_name),
      evaluate_model(testData$N, svm_test_pred, "SVM", "Test", method_name),
      
      evaluate_model(trainData$N, xgb_train_pred, "XGBoost", "Train", method_name),
      evaluate_model(testData$N, xgb_test_pred, "XGBoost", "Test", method_name),
      
      evaluate_model(trainData$N, enet_train_pred, "Elastic Net", "Train", method_name),
      evaluate_model(testData$N, enet_test_pred, "Elastic Net", "Test", method_name),
      
      evaluate_model(trainData$N, gbm_train_pred, "GBM", "Train", method_name),
      evaluate_model(testData$N, gbm_test_pred, "GBM", "Test", method_name),
      
      evaluate_model(trainData$N, stacked_train_pred, "Stacked Model", "Train", method_name),
      evaluate_model(testData$N, stacked_test_pred, "Stacked Model", "Test", method_name),
      
      evaluate_model(trainData$N, pls_svm_train_pred, "PLSR + SVM", "Train", method_name),
      evaluate_model(testData$N, pls_svm_test_pred, "PLSR + SVM", "Test", method_name)
    )
    
    all_train_preds[[method_name]] <- data.frame(
      Actual = trainData$N,
      PLSR = pls_train_pred,
      RandomForest = rf_train_pred,
      SVM = svm_train_pred,
      XGBoost = xgb_train_pred,
      ElasticNet = enet_train_pred,
      GBM = gbm_train_pred,
      StackedModel = stacked_train_pred,
      PLS_SVM = pls_svm_train_pred
    )
    
    all_test_preds[[method_name]] <- data.frame(
      Actual = testData$N,
      PLSR = pls_test_pred,
      RandomForest = rf_test_pred,
      SVM = svm_test_pred,
      XGBoost = xgb_test_pred,
      ElasticNet = enet_test_pred,
      GBM = gbm_test_pred,
      StackedModel = stacked_test_pred,
      PLS_SVM = pls_svm_test_pred
    )
  }, error = function(e) {
    cat("Error in preprocessing method:", method_name, "\nMessage:", e$message, "\n")
  })
}

# Print final results
print(all_results)

# Optional: Save to CSV
write.csv(all_results, "model_performance_summary.csv", row.names = FALSE)


# ---------------- Save All Predictions ----------------

# Combine training predictions
train_pred_all <- do.call(rbind, lapply(names(all_train_preds), function(preproc) {
  df <- all_train_preds[[preproc]]
  df$Preprocessing <- preproc
  df$Set <- "Train"
  df
}))

# Combine testing predictions
test_pred_all <- do.call(rbind, lapply(names(all_test_preds), function(preproc) {
  df <- all_test_preds[[preproc]]
  df$Preprocessing <- preproc
  df$Set <- "Test"
  df
}))

# Merge both
all_preds_combined <- rbind(train_pred_all, test_pred_all)

# Reorder columns for clarity
all_preds_combined <- all_preds_combined[, c("Preprocessing", "Set", "Actual",
                                             "PLSR", "RandomForest", "SVM", "XGBoost", 
                                             "ElasticNet", "GBM", "PLS_SVM")]

# Save to CSV
write.csv(all_preds_combined, "all_model_predictions.csv", row.names = FALSE)

write.csv(trainData, "train_data.csv", row.names = FALSE)
write.csv(testData, "test_data.csv", row.names = FALSE)
