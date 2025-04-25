# Load necessary libraries
library(nnet)       # For neural network modeling
library(survival)   # For survival analysis
library(survminer)  # For Kaplan–Meier plotting
library(ggplot2)    # For plot annotation

# ------------------------------
# 1. Data Loading & Preparation
# ------------------------------
df <- read.csv("~/OneDrive - Umich/Course/CHE 696 006/Final project/MUC/cluster_combine.csv", 
               stringsAsFactors = FALSE)

# Ensure Combined_Group is a factor (true risk labels)
df$Combined_Group <- as.factor(df$Combined_Group)
str(df)

# Define the neural network formula using predictor features
nn_formula <- Combined_Group ~ MUC1 + MUC2 + MUC4

# Set neural network parameters (adjust as desired)
nn_size   <- 2      # number of hidden units
nn_decay  <- 0.01   # weight decay (regularization)
nn_maxit  <- 500    # maximum iterations

# ------------------------------
# 2. Manual LOOCV using nnet
# ------------------------------
manual_loocv_nnet <- function(data, formula, size, decay, maxit) {
  n <- nrow(data)
  preds <- rep(NA, n)
  for (i in 1:n) {
    train_i <- data[-i, ]
    test_i  <- data[i, , drop = FALSE]
    # Train the neural network (classification mode)
    # Train the neural network (classification mode)
    model <- nnet(formula, data = train_i, size = size, decay = decay, maxit = maxit, trace = FALSE)
    # Predict for the left-out sample; type = "class" returns the predicted class
    preds[i] <- as.character(predict(model, newdata = test_i, type = "class"))
  }
  # Convert predictions to a factor with the same levels as the original Combined_Group
  preds <- factor(preds, levels = levels(data$Combined_Group))
  error_rate <- mean(preds != data$Combined_Group)
  return(list(predictions = preds, error = error_rate))
}

loocv_res_nn <- manual_loocv_nnet(df, nn_formula, size = nn_size, decay = nn_decay, maxit = nn_maxit)
cat("LOOCV Misclassification Error (Neural Network):", loocv_res_nn$error, "\n")
df$pred_label_LOOCV_NNet <- loocv_res_nn$predictions

# ------------------------------
# 3. Manual 10-Fold CV using nnet
# ------------------------------
manual_kfold_nnet <- function(data, formula, k = 10, size, decay, maxit) {
  set.seed(42)
  n <- nrow(data)
  folds <- sample(rep(1:k, length.out = n))
  preds <- rep(NA, n)
  for (i in 1:k) {
    test_idx <- which(folds == i)
    train_idx <- which(folds != i)
    train_data <- data[train_idx, ]
    test_data <- data[test_idx, , drop = FALSE]
    
    model <- nnet(formula, data = train_data, size = size, decay = decay, maxit = maxit, trace = FALSE)
    preds[test_idx] <- as.character(predict(model, newdata = test_data, type = "class"))
  }
  preds <- factor(preds, levels = levels(data$Combined_Group))
  error_rate <- mean(preds != data$Combined_Group)
  return(list(predictions = preds, error = error_rate))
}

kfold_res_nn <- manual_kfold_nnet(df, nn_formula, k = 10, size = nn_size, decay = nn_decay, maxit = nn_maxit)
cat("10-Fold CV Misclassification Error (Neural Network):", kfold_res_nn$error, "\n")
df$pred_label_kfold_NNet <- kfold_res_nn$predictions


# ------------------------------
# 4. Survival Analysis Based on LOOCV Predictions (Neural Network)
# ------------------------------
# Use LOOCV predictions for survival analysis.
cat("Predicted Label Distribution (LOOCV, NN):\n")
print(table(df$pred_label_LOOCV_NNet))

# Fit Kaplan–Meier survival curves using the full dataset and LOOCV NN predicted labels.
km_fit_nn <- survfit(Surv(Time_months, Status) ~ pred_label_LOOCV_NNet, data = df)

# Create legend labels with sample sizes (assuming levels are "1" and "2")
n_group1_nn <- sum(df$pred_label_LOOCV_NNet == "1")
n_group2_nn <- sum(df$pred_label_LOOCV_NNet == "2")
legend_labels_nn <- c(paste0("Group 1 (n=", n_group1_nn, ")"),
                      paste0("Group 2 (n=", n_group2_nn, ")"))

# ------------------------------
# 5. Cox Regression on LOOCV NN Predictions
# ------------------------------
cox_df_nn <- df[, c("Time_months", "Status", "pred_label_LOOCV_NNet")]
colnames(cox_df_nn) <- c("T", "E", "pred_label")
cox_model_nn <- coxph(Surv(T, E) ~ pred_label, data = cox_df_nn)
cox_summary_nn <- summary(cox_model_nn)
cat("\nCox Regression Summary (LOOCV NN Predictions):\n")
print(cox_summary_nn)

# Extract HR and CI from the Cox summary (using the conf.int table)
hr_nn    <- cox_summary_nn$conf.int[1, "exp(coef)"]
lower_nn <- cox_summary_nn$conf.int[1, "lower .95"]
upper_nn <- cox_summary_nn$conf.int[1, "upper .95"]

# Compute log-rank test p-value for LOOCV NN predictions
sdiff_nn <- survdiff(Surv(Time_months, Status) ~ pred_label_LOOCV_NNet, data = df)
logrank_p_nn <- 1 - pchisq(sdiff_nn$chisq, length(sdiff_nn$n) - 1)
cat("Log-rank test (LOOCV):", logrank_p_nn,"\n")

if (logrank_p_nn < 0.001) {
  logrank_text_nn <- "*P < 0.001 (log-rank test)"
} else {
  logrank_text_nn <- paste0("P = ", round(logrank_p_nn, 3), " (log-rank test)")
}

# log-rank for k-fold
sdiff_kfold <- survdiff(Surv(Time_months, Status) ~ pred_label_kfold_NNet, data = df)
logrank_p_kfold <- 1 - pchisq(sdiff_kfold$chisq, length(sdiff_kfold$n) - 1)
cat("Log-rank test (k-fold):", logrank_p_kfold,"\n")

annotation_text_nn <- paste0(
  "HR ", round(hr_nn, 2), " (", round(lower_nn, 2), "–", round(upper_nn, 2), ")\n",
  logrank_text_nn
)

# ------------------------------
# 6. Kaplan–Meier Plot for NN Predictions 
# ------------------------------
km_plot_nn <- ggsurvplot(
  km_fit_nn, data = df,
  pval = FALSE,              # Here we display the log-rank p-value (optional)
  conf.int = FALSE,         # no confidence intervals
  legend.labs = legend_labels_nn,
  legend.title = "",
  palette = c("blue", "red"),
  linetype = c("solid", "dashed"),
  title = "Neural Network",
  font.title = c(16, "plain", "black"),
  xlab = "Time (months)",
  ylab = "Survival Probability",
  risk.table = FALSE,
  font.legend = c(14, "plain", "black")
)

# Add custom annotation to the plot using ggplot2::annotate
km_plot_nn$plot <- km_plot_nn$plot +
  ggplot2::annotate("text",
                    x = 0.1,
                    y = 0.12,
                    label = annotation_text_nn,
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
ggsave("OneDrive - Umich/Course/CHE 696 006/Final project/R/NNet_loocv.png", plot = km_plot_nn$plot, width = 7, height = 6, units = "in", dpi = 300)
# Print the final Kaplan–Meier plot
print(km_plot_nn)