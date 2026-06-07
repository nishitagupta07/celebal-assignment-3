# HELP International Country Clustering

This project uses unsupervised machine learning to group countries by socio-economic condition and identify which countries should be prioritized for HELP International aid.

## Project Files
- `model.py`: main analysis and clustering script
- `Country-data.csv`: input dataset
- `submission.md`: ready-to-submit written summary
- `.dist/`: generated plots and exported output files

## What the Script Does
- Loads and cleans the country dataset
- Fills missing numeric values with medians
- Visualizes correlations and feature distributions
- Selects `income` and `gdpp` for clustering
- Scales data with `RobustScaler`
- Uses PCA for cluster visualization
- Finds the best K-Means cluster count using silhouette score
- Runs DBSCAN as a comparison model
- Shortlists countries from the lowest-income cluster

## Generated Outputs
When you run `model.py`, it saves the following inside `.dist/`:
- `correlation_heatmap.png`
- `boxplot_*.png`
- `elbow_method.png`
- `pca_clusters.png`
- `cluster_profile.csv`
- `help_international_submission.csv`
- `model_summary.txt`

## Result Summary
- Best K-Means clusters: `2`
- Silhouette score: `0.7464`
- Priority cluster: the cluster with the lowest average income

The top recommended countries include Congo (Dem. Rep.), Liberia, Burundi, Niger, Central African Republic, Mozambique, and Malawi.



```bash
pip install pandas numpy matplotlib scikit-learn seaborn
