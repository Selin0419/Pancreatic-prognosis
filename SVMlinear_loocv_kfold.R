# Load necessary libraries
library(e1071)      # For SVM
library(survival)   # For survival analysis
library(survminer)  # For Kaplan–Meier plotting
library(ggplot2)    # For annotation if needed
library(kernlab)

# ------------------------------
# 1. Data Loading & Preparation
# ------------------------------
df <- read.csv("~/OneDrive - Umich/Course/CHE 696 006/Final project/MUC/cluster_combine.csv", 
               stringsAsFactors = FALSE)

# Ensure the risk label is a factor (this is the true label from clustering)
df$Combined_Group <- as.factor(df$Combined_Group)
str(df)  # Verify the structure

# Define the SVM formula using the predictor features
svm_formula <- Combined_Group ~ MUC1 + MUC2 + MUC4
C = 2

# ------------------------------
# 2. Manual LOOCV using RBF Kernel (No data split)
# ------------------------------
# kpar: sigma. For gamma similar to "scale" in Python, you might compute it as:
X <- as.matrix(df[, c("MUC1", "MUC2", "MUC4")])
n_features <- ncol(X)
# Compute the average variance of each feature
avg_var <- mean(apply(X, 2, var))
# Set gamma similar to Python's gamma = "scale"
gamma_value <- 1 / (n_features * avg_var)
cat("Calculated gamma:", gamma_value, "\n")

# ------------------------------
# 2. Manual 10-Fold CV Function for SVM with RBF Kernel
# ------------------------------
manual_kfold <- function(data, formula, k = 10, C = 2, gamma = gamma_value) {
  set.seed(42)
  n <- nrow(data)
  # Create fold assignments (not stratified in this simple version)
  folds <- sample(rep(1:k, length.out = n))
  preds <- rep(NA, n)
  for(i in 1:k) {
    test_idx <- which(folds == i)
    train_idx <- which(folds != i)
    train_data <- data[train_idx, ]
    test_data <- data[test_idx, , drop = FALSE]
    
    # Train SVM on the training folds using RBF kernel
    model <- ksvm(formula, data = train_data, kernel = "vanilladot", C = C, gamma = gamma_value)
    
    # Predict for the left-out fold; convert predictions to character
    preds[test_idx] <- as.character(predict(model, newdata = test_data))
  }
  # Convert predictions to a factor with the same levels as the original label
  preds <- factor(preds, levels = levels(data$Combined_Group))
  error_rate <- mean(preds != data$Combined_Group)
  return(list(predictions = preds, error = error_rate))
}

#kfold CV using RBF
kfold_res <- manual_kfold(df, svm_formula, k = 10, C = 2, gamma = gamma_value)
cat("10-fold CV Misclassification Error:", kfold_res$error, "\n")

# Save the k-fold predictions in the data frame
df$pred_label_kfold <- kfold_res$predictions

manual_loocv_rbf <- function(data, formula, cost = 2, gamma = gamma_value) {
  n <- nrow(data)
  preds <- rep(NA, n)
  for(i in 1:n) {
    train_i <- data[-i, ]
    test_i  <- data[i, , drop = FALSE]
    # Train SVM with RBF kernel
    model <- ksvm(formula, data = train_i, kernel = "vanilladot", C = C, gamma = gamma_value)
    
    # Predict for the left-out sample and convert to character
    preds[i] <- as.character(predict(model, newdata = test_i))
  }
  # Convert predictions to a factor with the same levels as the original label
  preds <- factor(preds, levels = levels(data$Combined_Group))
  error_rate <- mean(preds != data$Combined_Group)
  return(list(predictions = preds, error = error_rate))
}

loocv_res <- manual_loocv_rbf(df, svm_formula, cost = 2, gamma = gamma_value)
cat("LOOCV Misclassification Error:", loocv_res$error, "\n")

# Save the LOOCV predictions in the dataframe
df$pred_label_LOOCV <- loocv_res$predictions

# ------------------------------
# 3. Survival Analysis Based on LOOCV Predictions
# ------------------------------
# Use the LOOCV predicted labels for survival analysis.
# (Ensure that the column names match your data; here we assume "Time_months" and "Status".)
# Compute the log-rank test p-value using survdiff()
sdiff_loocv <- survdiff(Surv(Time_months, Status) ~ pred_label_LOOCV, data = df)
logrank_p_loocv <- 1 - pchisq(sdiff_loocv$chisq, length(sdiff_loocv$n) - 1)

if (logrank_p_loocv < 0.001) {
  logrank_text <- "*P < 0.001 (log-rank test)"
} else {
  logrank_text <- paste0("P = ", round(logrank_p_loocv, 3), " (log-rank test)")
}

sdiff_kfold <- survdiff(Surv(Time_months, Status) ~ pred_label_kfold, data = df)
logrank_p_kfold <- 1 - pchisq(sdiff_kfold$chisq, length(sdiff_kfold$n) - 1)
cat("Log-rank test:", logrank_p_kfold, "\n")

# Fit a Cox proportional hazards model to obtain the hazard ratio (HR)
cox_model <- coxph(Surv(Time_months, Status) ~ pred_label_LOOCV, data = df)
cox_summary <- summary(cox_model)
hr    <- cox_summary$conf.int[1, "exp(coef)"]
lower <- cox_summary$conf.int[1, "lower .95"]
upper <- cox_summary$conf.int[1, "upper .95"]
cat("\nCox Regression Summary (LOOCV Predictions):\n")
print(cox_summary)

# Prepare data for Cox regression: use the predicted labels from k-fold CV.
cox_df_kfold <- df[, c("Time_months", "Status", "pred_label_kfold")]
colnames(cox_df_kfold) <- c("T", "E", "pred_label")
cox_model_kfold <- coxph(Surv(T, E) ~ pred_label, data = cox_df_kfold)
cox_summary_kfold <- summary(cox_model_kfold)
cat("\nCox Regression Summary (10-fold CV Predictions):\n")
print(cox_summary_kfold)

# Format the hazard ratio text (we won't include the Cox p-value in the annotation)
annotation_text <- paste0(
  "HR ", round(hr, 2),
  " (", round(lower, 2), "–", round(upper, 2), ")\n",
  "P = ", formatC(logrank_p_loocv, format="f", digits=3), " (log-rank test)"
)

# ------------------------------
# 4. Kaplan–Meier Plot
# ------------------------------
# Fit Kaplan–Meier curves grouped by the LOOCV predictions.
km_fit <- survfit(Surv(Time_months, Status) ~ pred_label_LOOCV, data = df)

# Create legend labels with sample sizes (assuming the levels are "1" and "2")
n_group1 <- sum(df$pred_label_LOOCV == "1")
n_group2 <- sum(df$pred_label_LOOCV == "2")
legend_labels <- c(paste0("Group 1 (n=", n_group1, ")"),
                   paste0("Group 2 (n=", n_group2, ")"))

# Create the Kaplan–Meier survival plot.
# We set pval = FALSE because we'll annotate with our own p-value text.
km_plot <- ggsurvplot(km_fit, data = df,
                      pval = FALSE,
                      conf.int = FALSE,
                      legend.labs = legend_labels,
                      legend.title = "",
                      palette = c("blue", "red"),
                      linetype = c("solid", "dashed"),
                      title = "SVM Linear",
                      font.title = c(16, "plain", "black"),
                      xlab = "Time (months)",
                      ylab = "Survival Probability",
                      risk.table = FALSE,
                      font.legend = c(14, "plain", "black"))

# ------------------------------
# 5. Annotate the KM Plot
# ------------------------------
# Add our custom annotation (HR and log-rank p-value) using ggplot2::annotate().
km_plot$plot <- km_plot$plot +
  ggplot2::annotate("text",
                    x = 0.1,  # adjust position as needed
                    y = 0.12,
                    label = annotation_text,
                    size = 5,
                    color = "black",
                    fontface = "plain",
                    hjust = 0, 
                    vjust = 1)+
  theme(
    plot.title = element_text(hjust = 0.5),
    # Put the legend on the right side
    legend.position = c(0.7, 0.75)        
  ) +
  # Make the legend 2 rows in a single column
  guides(
    color    = guide_legend(ncol = 1),
    linetype = guide_legend(ncol = 1)
  )
ggsave("SVM_linear_loocv.png", plot = km_plot$plot, width = 7, height = 6, units = "in", dpi = 300)
# Print the final Kaplan–Meier plot
print(km_plot)