# Load necessary libraries
library(caret)       # for stratified data partition
library(e1071)       # for SVM modeling
library(survival)    # for survival analysis (Surv, survfit, coxph)
library(survminer)   # for Kaplan–Meier plotting
library(ggplot2)     # for custom plot annotations
library(kernlab)

# ------------------------------
# 1. Data Loading & Preparation
# ------------------------------
df <- read.csv("~/OneDrive - Umich/Course/CHE 696 006/Final project/MUC/cluster_combine.csv", 
               stringsAsFactors = FALSE)
# Ensure the risk label is a factor
df$Combined_Group <- as.factor(df$Combined_Group)
str(df)

# kpar: sigma. For gamma similar to "scale" in Python, you might compute it as:
X <- as.matrix(df[, c("MUC1", "MUC2", "MUC4")])
n_features <- ncol(X)
# Compute the average variance of each feature
avg_var <- mean(apply(X, 2, var))
# Set gamma similar to Python's gamma = "scale"
gamma_value <- 1 / (n_features * avg_var)
cat("Calculated gamma:", gamma_value, "\n")


# ------------------------------
# 2. Split the Data (Stratified)
# ------------------------------
set.seed(42)
# Use createDataPartition to stratify by Combined_Group (60% training, 40% test)
train_index <- createDataPartition(df$Combined_Group, p = 0.6, list = FALSE)
train_data <- df[train_index, ]
test_data  <- df[-train_index, ]

cat("Training set distribution:\n")
print(prop.table(table(train_data$Combined_Group)))
cat("\nTest set distribution:\n")
print(prop.table(table(test_data$Combined_Group)))

# ------------------------------
# 3. SVM Model Training using RBF Kernel
# ------------------------------
# Define the SVM prediction formula
svm_formula <- Combined_Group ~ MUC1 + MUC2 + MUC4

# Train SVM with radial basis function (RBF) kernel
# (Adjust cost and gamma as needed; here we use cost = 1 and gamma = 0.5 as an example)
svm_rbf <- ksvm(svm_formula, data = train_data, kernel = "rbfdot", 
                kpar = list(sigma = gamma_value), C = 2)

# ------------------------------
# 4. Predict on the Test Set
# ------------------------------
predicted <- predict(svm_rbf, newdata = test_data)
print(table(predicted))
# Ensure predicted labels are factors with the same levels as Combined_Group
test_data$pred_label <- factor(predicted, levels = levels(df$Combined_Group))
head(test_data$pred_label)

# ------------------------------
# 5. Survival Analysis on Test Set Predictions
# ------------------------------
# Create a survival object from the test set
surv_obj <- Surv(test_data$Time_months, test_data$Status)

# Fit Kaplan–Meier curves grouped by the SVM predictions
km_fit <- survfit(surv_obj ~ pred_label, data = test_data)

# Perform log-rank test
logrank_res <- survdiff(surv_obj ~ pred_label, data = test_data)
p_val <- 1 - pchisq(logrank_res$chisq, length(logrank_res$n) - 1)
if (p_val < 0.001) {
  p_str <- "*P < 0.001 (log-rank test)"
} else {
  p_str <- paste0("P = ", formatC(p_val, format = "f", digits = 3), " (log-rank test)")
}

# Fit a Cox proportional hazards model to estimate hazard ratio (HR)
cox_model <- coxph(surv_obj ~ pred_label, data = test_data)
cox_summary <- summary(cox_model)
hr_est <- cox_summary$conf.int[1, "exp(coef)"]
lower   <- cox_summary$conf.int[1, "lower .95"]
upper   <- cox_summary$conf.int[1, "upper .95"]
hr_str <- paste0("HR ", round(hr_est, 2), " (", round(lower, 2), "–", round(upper, 2), ")")
annotation_text <- paste0(hr_str, "\n", p_str)

# Calculate the confusion matrix between predicted labels and true labels
conf_mat <- table(Predicted = predicted, True = test_data$Combined_Group)
print(conf_mat)

# Calculate misclassification error:
# misclassification error = 1 - (number of correct predictions)/(total number of samples)
misclassification_error <- 1 - sum(diag(conf_mat)) / sum(conf_mat)
cat("Misclassification Error:", round(misclassification_error, 3), "\n")


# ------------------------------
# 6. Kaplan–Meier Plot with Annotation
# ------------------------------
# Create legend labels that include sample sizes for each predicted group
n_group1 <- sum(test_data$pred_label == "1")
n_group2 <- sum(test_data$pred_label == "2")
legend_labels <- c(paste0("Group 1 (n=", n_group1, ")"),
                   paste0("Group 2 (n=", n_group2, ")"))

km_plot <- ggsurvplot(km_fit, data = test_data,
                      pval = FALSE,
                      conf.int = FALSE,
                      legend.labs = legend_labels,
                      legend.title = "",
                      palette = c("blue", "red"),
                      linetype = c("solid", "dashed"),
                      title = "Support Vector Machine (rbf)",
                      font.title = c(16, "plain", "black"),
                      xlab = "Time (months)",
                      ylab = "Survival Probability",
                      risk.table = FALSE,
                      font.legend = c(14, "plain", "black"))

# Add custom annotation (HR and log-rank p-value) using ggplot2::annotate
km_plot$plot <- km_plot$plot +
  ggplot2::annotate("text",
                    x = 0.1,  # adjust as needed
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
ggsave("SVM_rbf_performance.png", plot = km_plot$plot, width = 7, height = 6, units = "in", dpi = 300)
# Print the final Kaplan–Meier plot
print(km_plot)