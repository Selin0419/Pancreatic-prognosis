# Install if needed:
#install.packages("ComplexHeatmap")
#install.packages("circlize")
library(ComplexHeatmap)
library(circlize)
library(pheatmap)
library(grid)

legend_breaks <- c(-5, 0, 7)
legend_colors <- c("blue", "white", "red")
# ------------------------------
# 1. Data Loading & Preparation
# ------------------------------
df <- read.csv("~/OneDrive - Umich/Course/CHE 696 006/Final project/MUC/cluster_combine.csv", 
               stringsAsFactors = FALSE)

# ------------------------------
# 2. Select Only the Numeric Columns for Clustering
# ------------------------------
# Create a new data frame (or matrix) that contains only the columns used for clustering.
# Perform hierarchical clustering on the numeric data (MUC1, MUC4)
df_numeric <- df[, c("MUC1", "MUC4")]
d <- dist(df_numeric, method = "euclidean")
hc <- hclust(d, method = "ward.D2")
cluster_labels <- cutree(hc, k = 3)
df$cluster_original <- cluster_labels

# Combine clusters: designate cluster 1 as "Cluster 1" and the rest as "Other clusters"
df$final_cluster <- ifelse(df$cluster_original == 1, "Cluster 3", "Cluster 1 and 2")
print(df[, c("MUC1", "MUC4", "cluster_original", "final_cluster")])

mat <- as.matrix(df_numeric)
mat_t <- t(mat)

data_range <- range(mat_t, na.rm = TRUE)
if(data_range[1] == data_range[2]){
  # If the data is constant, create a small buffer
  data_range <- c(data_range[1] - 0.1, data_range[2] + 0.1)
}

# ------------------------------
# 2. Compute Hierarchical Clustering for Columns (Samples)
# ------------------------------
# Compute the distance between samples (columns of mat_t)
d <- dist(t(mat_t), method = "euclidean")
# Perform hierarchical clustering using Ward's method
hc <- hclust(d, method = "ward.D2")
# Convert the hclust object to a dendrogram for ComplexHeatmap
col_dend <- as.dendrogram(hc)


# -------------------------------------
# 3. Determine Clusters and Combine Them
# -------------------------------------
# Cut the dendrogram into 3 clusters.
hc <- hclust(d, method = "ward.D2")
col_clusters <- cutree(hc, k = 3)
# Combine clusters: we assume that cluster labeled 1 remains as "Cluster 1",
# and clusters labeled 2 and 3 are merged into "Other clusters".
col_clusters_factor <- factor(ifelse(col_clusters == 1, "Cluster 3", "Cluster 1 and 2"),
                              levels = c("Cluster 3", "Cluster 1 and 2"))
# (Check the distribution:)
print(table(col_clusters_factor))

# -------------------------------------
# 4. Create a Bottom Annotation for the Final Clusters
# -------------------------------------
# Using the cluster factor for the columns (samples), create a HeatmapAnnotation.
# We will hide its legend.
# Relabel factor levels
col_clusters_renamed <- factor(
  ifelse(col_clusters == 1, "High-risk", "Low-risk"),
  levels = c("High-risk", "Low-risk")
)

ha_bottom <- HeatmapAnnotation(
  Cluster = col_clusters_renamed,
  which = "column",
  annotation_name_side = "left",  # For column annotations, name position must be "left" or "right".
  col = list(Cluster = c("High-risk" = "black", "Low-risk" = "grey")),
  annotation_height = unit(1, "pt")
)

# -------------------------------------
# 5. Draw the Heatmap Using ComplexHeatmap
# -------------------------------------
# Here, we want:
#   - Rows: genes (MUC1, MUC4)
#   - Columns: samples
#   - No row clustering (so no dendrogram on the left)
#   - A precomputed column dendrogram (with the samples)
#   - The columns split into 2 groups (based on our final clusters)
#   - A bottom annotation bar for the clusters
#   - The heatmap colored with blue-white-red palette.
ht <- Heatmap(
  matrix = mat_t,
  name = "Value",  # This will be renamed in the legend as needed; e.g., "Hypomethylation"
  cluster_rows = FALSE,         # Do not cluster rows (genes)
  cluster_columns = col_dend,   # Use our computed dendrogram for columns (samples)
  #column_split = 2,   # Split the columns into 2 groups; note, this uses the dendrogram,
  # so it splits the tree into 2 parts based on height.
  column_title_side = "bottom", # Display split labels under the heatmap
  bottom_annotation = ha_bottom,  # Attach our bottom annotation bar.
  show_column_names = FALSE,    # Hide sample IDs for clarity.
  row_names_side = "left",      # Display gene names on the left.
  heatmap_legend_param = list(
    title = "Value",
    at = legend_breaks,            # should have the same length as
    labels = as.character(legend_breaks)  # these labels
  ),
  col = colorRampPalette(c("blue", "white", "red"))(100)
  # Here’s the key: override the default slice title function
)
# -------------------------------------
# 6. Save the Figure at 300 dpi
# -------------------------------------
# Open a PNG device specifying size and resolution.
png("heatmap_output.png", width = 8, height = 3, units = "in", res = 300)
draw(ht, heatmap_legend_side = "right", annotation_legend_side = "right")
dev.off()