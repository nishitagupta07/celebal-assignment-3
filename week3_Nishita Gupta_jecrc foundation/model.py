import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

if sns is not None:
    sns.set(style="whitegrid")

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Country-data.csv"
OUTPUT_DIR = BASE_DIR / ".dist"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def plot_heatmap(dataframe, filename):
    plt.figure(figsize=(12, 8))

    if sns is not None:
        sns.heatmap(dataframe, annot=True, cmap="coolwarm", fmt=".2f")
    else:
        ax = plt.gca()
        heatmap = ax.imshow(dataframe, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(dataframe.columns)))
        ax.set_yticks(range(len(dataframe.columns)))
        ax.set_xticklabels(dataframe.columns, rotation=45, ha="right")
        ax.set_yticklabels(dataframe.columns)

        for row in range(len(dataframe.columns)):
            for col in range(len(dataframe.columns)):
                ax.text(col, row, f"{dataframe.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)

        plt.colorbar(heatmap)

    plt.title("Correlation Heatmap")
    save_plot(filename)


def plot_boxplot(series, column_name, filename):
    plt.figure(figsize=(8, 3))

    if sns is not None:
        sns.boxplot(x=series)
    else:
        plt.boxplot(series, vert=False)

    plt.title(f"Boxplot: {column_name}")
    save_plot(filename)


def plot_scatter(viz_df, filename):
    plt.figure(figsize=(8, 6))

    if sns is not None:
        sns.scatterplot(data=viz_df, x="pca1", y="pca2", hue="cluster", palette="tab10")
    else:
        scatter = plt.scatter(viz_df["pca1"], viz_df["pca2"], c=viz_df["cluster"], cmap="tab10", s=60)
        handles, labels = scatter.legend_elements()
        plt.legend(handles, labels, title="cluster")

    plt.title("K-Means Clusters Visualized with PCA")
    save_plot(filename)



print("Using dataset:", DATA_FILE)
print("Seaborn available:", sns is not None)


# Load dataset locally 
df = pd.read_csv(DATA_FILE)


# Basic inspection
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include="all").T)
print("Columns:", df.columns.tolist())


# Cleaning and preprocessing
df = df.copy()
df.columns = [c.strip().lower() for c in df.columns]
df = df.drop_duplicates()

for col in df.columns:
    if col != "country":
        df[col] = pd.to_numeric(df[col], errors="coerce")

numeric_cols = [c for c in df.columns if c != "country"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print("\nMissing values after cleaning:")
print(df.isna().sum())


# Correlation heatmap
plot_heatmap(df.select_dtypes(include=np.number).corr(), "correlation_heatmap.png")


# Boxplots
for col in numeric_cols:
    plot_boxplot(df[col], col, f"boxplot_{col}.png")


# Feature selection and scaling
selected_features = ["income", "gdpp"]
features = df[selected_features].copy()
scaler = RobustScaler()
X_scaled = scaler.fit_transform(features)

print("\nSelected features for clustering:", selected_features)
print("\nScaled feature shape:", X_scaled.shape)


# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Feature shape after PCA:", X_pca.shape)


# Elbow method
inertias = []
k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=30)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(8, 4))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
save_plot("elbow_method.png")


# KMeans clustering
best_k = list(k_values)[int(np.argmax(silhouette_scores))]
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=30)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

best_silhouette = silhouette_score(X_scaled, df["kmeans_cluster"])

print("\nBest k based on silhouette:", best_k)
print("Silhouette Score:", best_silhouette)

if "country" in df.columns:
    print(df[["country", "kmeans_cluster"]].head())
else:
    print(df[["kmeans_cluster"]].head())


# DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

print("\nDBSCAN cluster counts:")
print(df["dbscan_cluster"].value_counts().sort_index())


# PCA visualization
viz = pd.DataFrame(
    {
        "pca1": X_pca[:, 0],
        "pca2": X_pca[:, 1],
        "cluster": df["kmeans_cluster"],
    }
)

plot_scatter(viz, "pca_clusters.png")


# Cluster profiling
profile = df.groupby("kmeans_cluster")[numeric_cols].mean().round(2)
profile.to_csv(OUTPUT_DIR / "cluster_profile.csv")

print("\nCluster profile:")
print(profile)


# HELP International shortlisting
poor_cluster = df.groupby("kmeans_cluster")["income"].mean().idxmin()
help_countries = df[df["kmeans_cluster"] == poor_cluster]

result = help_countries[["country", "income", "gdpp", "child_mort", "life_expec"]]
result = result.sort_values(by="income")
top_20_result = result.head(20).reset_index(drop=True)
top_20_result.to_csv(OUTPUT_DIR / "help_international_submission.csv", index=False)

print("\nCountries that should receive HELP International Aid:")
print(top_20_result)

print("\nModel Summary:")
print("Best k:", best_k)
print(f"Silhouette Score: {best_silhouette:.4f}")
print(f"\nPlots saved to: {OUTPUT_DIR}")

summary_lines = [
    "HELP International Country Clustering Summary",
    f"Dataset: {DATA_FILE.name}",
    f"Selected features: {', '.join(selected_features)}",
    f"Best k (K-Means): {best_k}",
    f"Silhouette Score: {best_silhouette:.4f}",
    "",
    "Top 20 countries recommended for aid:",
]
summary_lines.extend(
    [
        f"{row.country}: income={row.income}, gdpp={row.gdpp}, child_mort={row.child_mort}, life_expec={row.life_expec}"
        for row in top_20_result.itertuples(index=False)
    ]
)

(OUTPUT_DIR / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
