# Load necessary libraries
library(caret)       # for stratified data partition
library(nnet)        # for neural network modeling
library(survival)    # for survival analysis (Surv, survfit, coxph)
library(survminer)   # for Kaplan–Meier plotting
library(ggplot2)     # for annotation

# ------------------------------
# 1. Data Loading & Preparation
# ------------------------------
df <- read.csv("~/OneDrive - Umich/Course/CHE 696 006/Final project/MUC/cluster_combine.csv", 
               stringsAsFactors = FALSE)

# Ensure the risk label is a factor
df$Combined_Group <- as.factor(df$Combined_Group)
str(df)

# ------------------------------
# 2. Split the Data (Stratified)
# ------------------------------
set.seed(42)
# Randomly reorder the entire dataset first
df_shuffled <- df[sample(nrow(df)), ]

# Then partition
train_index <- createDataPartition(df_shuffled$Combined_Group, p = 0.6, list = FALSE)
train_data <- df_shuffled[train_index, ]
test_data  <- df_shuffled[-train_index, ]

# Confirm stratification
cat("Training set distribution:\n")
print(prop.table(table(train_data$Combined_Group)))
cat("\nTest set distribution:\n")
print(prop.table(table(test_data$Combined_Group)))

# ------------------------------
# 3. Neural Network Model Training (nnet)
# ------------------------------
# Define the prediction formula
nn_formula <- Combined_Group ~ MUC1 + MUC2 + MUC4

# Set neural network parameters: 
# size = number of hidden units; decay = weight decay; maxit = maximum iterations.
nn_size   <- 2
nn_decay  <- 0.01    
nn_maxit  <- 500    # you can increase this if needed

# Train the neural network on the training set
nnet_model <- nnet(nn_formula, data = train_data, size = nn_size, decay = nn_decay, 
                   maxit = nn_maxit, trace = FALSE)

# ------------------------------
# 4. Predict on the Test Set            
# ------------------------------
# Predict the class labels on the test set using type = "class"
predicted <- predict(nnet_model, newdata = test_data, type = "class")
print(table(predicted))

# Convert predictions to a factor with levels matching the true labels (assumed to be "1" and "2")
test_data$pred_label <- factor(predicted, levels = levels(df$Combined_Group))
head(test_data$pred_label)

# ------------------------------
# 5. Survival Analysis on Test Set Predictions
# ------------------------------
# Create a survival object from the test set
surv_obj <- Surv(test_data$Time_months, test_data$Status)

# Fit Kaplan–Meier curves grouped by the predicted labels from the NN
km_fit <- survfit(surv_obj ~ pred_label, data = test_data)

# Perform log-rank test
logrank_res <- survdiff(surv_obj ~ pred_label, data = test_data)
p_val <- 1 - pchisq(logrank_res$chisq, length(logrank_res$n) - 1)
if (p_val < 0.001) {
  p_str <- "*P < 0.001 (log-rank test)"
} else {
  p_str <- paste0("P = ", formatC(p_val, format="f", digits=3), " (log-rank test)")
}

# Fit a Cox proportional hazards model to estimate the hazard ratio (HR)
cox_model <- coxph(surv_obj ~ pred_label, data = test_data)
cox_summary <- summary(cox_model)
hr_est <- cox_summary$conf.int[1, "exp(coef)"]
lower   <- cox_summary$conf.int[1, "lower .95"]
upper   <- cox_summary$conf.int[1, "upper .95"]
hr_str <- paste0("HR ", round(hr_est,2), " (", round(lower,2), "–", round(upper,2), ")")
annotation_text <- paste0(hr_str, "\n", p_str)

# ------------------------------
# 6. Kaplan–Meier Plot with Annotation
# ------------------------------
# Create legend labels that include sample sizes for each predicted group
n_group1 <- sum(test_data$pred_label == "1")
n_group2 <- sum(test_data$pred_label == "2")
legend_labels <- c(paste0("Group 1 (n=", n_group1, ")"),
                   paste0("Group 2 (n=", n_group2, ")"))

km_plot <- ggsurvplot(
  km_fit, data = test_data,
  pval = FALSE,             # We'll annotate with our custom p-value
  conf.int = FALSE,
  legend.labs = legend_labels,
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

# Add custom annotation with HR and log-rank p-value using ggplot2::annotate
km_plot$plot <- km_plot$plot +
  ggplot2::annotate("text",
                    x = 0.1,
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
ggsave("NNet_performance.png", plot = km_plot$plot, width = 7, height = 6, units = "in", dpi = 300)
# Print the final Kaplan–Meier plot
print(km_plot)

# Calculate the confusion matrix between predicted labels and true labels
conf_mat <- table(Predicted = predicted, True = test_data$Combined_Group)
print(conf_mat)

# Calculate misclassification error:
# misclassification error = 1 - (number of correct predictions)/(total number of samples)
misclassification_error <- 1 - sum(diag(conf_mat)) / sum(conf_mat)
cat("Misclassification Error:", round(misclassification_error, 3), "\n")
