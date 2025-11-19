"""
Author: Ibrahim Olayinka Abdulsalam
Clustering and Fitting Assignment 
Completed code using the Used Car Price Prediction Dataset.
Dataset renamed to data.csv as instructed.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

# additional imports needed for clustering/fitting
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


def plot_relational_plot(df):
    """Create and save a relational scatter plot of milage vs price."""
    fig, ax = plt.subplots(figsize=(10, 6))

    xcol = "milage" if "milage" in df.columns else "mileage"
    ycol = "price"

    if xcol not in df.columns or ycol not in df.columns:
        plt.close(fig)
        return

    data = df[[xcol, ycol]].dropna()
    if data.shape[0] == 0:
        ax.text(0.5, 0.5, "No data available", ha="center")
        plt.tight_layout()
        plt.savefig("relational_plot.png")
        plt.close(fig)
        return

    sns.scatterplot(data=data, x=xcol, y=ycol, ax=ax, alpha=0.7)
    ax.set_title(f"{ycol} vs {xcol}")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)

    plt.tight_layout()
    plt.savefig("relational_plot.png")
    plt.close(fig)
    return

    hue = "transmission" if "transmission" in df.columns else None
    sns.scatterplot(data=df, x=xcol, y=ycol, hue=hue, ax=ax, alpha=0.7)

    ax.set_title(f"{ycol} vs {xcol}")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)

    if hue:
        ax.legend(loc="best", fontsize="small")

    plt.savefig("relational_plot.png")
    plt.close(fig)
    return


def plot_categorical_plot(df):
    """Create and save a categorical plot: average price by fuel type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if "fuel_type" in df.columns and "price" in df.columns:
        agg = df.groupby("fuel_type")["price"].mean().sort_values(ascending=False)

        if agg.shape[0] == 0:
            ax.text(0.5, 0.5, "No fuel_type data", ha="center")
        else:
            sns.barplot(x=agg.index, y=agg.values, ax=ax)
            plt.setp(ax.get_xticklabels(), rotation=0)

        ax.set_title("Average price by fuel type")
        ax.set_xlabel("Fuel type")
        ax.set_ylabel("Average price")

    else:
        ax.text(0.5, 0.5, "fuel_type or price missing", ha="center")

    plt.tight_layout()
    plt.savefig("categorical_plot.png")
    plt.close(fig)
    return

    
def plot_statistical_plot(df):
    """Create and save a statistical plot (correlation heatmap + boxplot)."""
    numeric = df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(10, 6))
    if numeric.shape[1] >= 2:
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
    else:
        ax.text(0.5, 0.5, "Not enough numeric columns", ha="center")

    plt.tight_layout()
    plt.savefig("statistical_plot.png")
    plt.close(fig)

    
def statistical_analysis(df, col: str):
    """Compute mean, stddev, skewness, and excess kurtosis for a column."""
    if col not in df.columns:
        numcols = df.select_dtypes(include=[np.number]).columns
        if len(numcols) == 0:
            raise ValueError("No numeric columns available for analysis.")
        col = numcols[0]

    series = df[col].dropna().astype(float)
    mean = float(series.mean())
    stddev = float(series.std(ddof=0))
    skew = float(ss.skew(series))
    excess_kurtosis = float(ss.kurtosis(series, fisher=True))

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Preprocess dataset and standardize column names / extract numeric values."""
    df = df.copy()

    # map original data columns
    colmap = {}
    if "price" in df.columns:
        colmap["price"] = "price"
    if "milage" in df.columns:
        colmap["milage"] = "mileage"
    if "model_year" in df.columns:
        colmap["model_year"] = "year"
    if "engine" in df.columns:
        colmap["engine"] = "engine"
    if "fuel_type" in df.columns:
        colmap["fuel_type"] = "fuel_type"
    if "transmission" in df.columns:
        colmap["transmission"] = "transmission"
    if "accident" in df.columns:
        colmap["accident"] = "accident"
    if "clean_title" in df.columns:
        colmap["clean_title"] = "clean_title"
    if "brand" in df.columns:
        colmap["brand"] = "brand"
    if "model" in df.columns:
        colmap["model"] = "model"

    df = df.rename(columns=colmap)

    # fix price
    if "price" in df.columns:
        df["price"] = df["price"].astype(str).str.replace("$", "").str.replace(",", "")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # fix mileage
    if "mileage" in df.columns:
        df["mileage"] = df["mileage"].astype(str).str.replace(",", "")
        df["mileage"] = df["mileage"].str.extract(r"(\d+\.?\d*)").astype(float)

    # fix year
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["age"] = 2025 - df["year"]

    # engine extraction
    if "engine" in df.columns:
        df["engine_cc"] = df["engine"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)

    df = df.dropna(subset=["price", "mileage", "age"], how="any")
    return df


def writing(moments, col):
    print(f"For the attribute {col}:")
    print(
        f"Mean = {moments[0]:.2f}, "
        f"Standard Deviation = {moments[1]:.2f}, "
        f"Skewness = {moments[2]:.2f}, and "
        f"Excess Kurtosis = {moments[3]:.2f}."
    )
    return


def perform_clustering(df, col1, col2):
    def plot_elbow_method(inertias):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ks = list(range(2, 2 + len(inertias)))
        ax.plot(ks, inertias, marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow plot")
        plt.savefig("elbow_plot.png")
        plt.close(fig)
        return

    def one_silhouette_inertia(data_scaled):
        best_k = 2
        best_score = -1
        inertias = []
        for k in range(2, 7):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(data_scaled)
            inertias.append(km.inertia_)
            try:
                score = silhouette_score(data_scaled, labels)
            except Exception:
                score = -1
            if score > best_score:
                best_score = score
                best_k = k
        return best_k, best_score, inertias

    # Gather data and scale
    if col1 not in df.columns or col2 not in df.columns:
        numcols = df.select_dtypes(include=[np.number]).columns
        if len(numcols) >= 2:
            col1, col2 = numcols[0], numcols[1]
        else:
            raise ValueError("Not enough numeric columns for clustering.")

    data_df = df[[col1, col2]].dropna().astype(float)
    if data_df.shape[0] == 0:
        raise ValueError("No rows with both clustering columns present.")
    data = data_df.values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Find best number of clusters
    best_k, best_score, inertias = one_silhouette_inertia(data_scaled)
    plot_elbow_method(inertias)

    # Fit KMeans with chosen k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    centers = kmeans.cluster_centers_
    centers_unscaled = scaler.inverse_transform(centers)
    xkmeans = centers_unscaled[:, 0]
    ykmeans = centers_unscaled[:, 1]
    centre_labels = [f"c{idx}" for idx in range(len(xkmeans))]

    return labels, data, xkmeans, ykmeans, centre_labels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    fig, ax = plt.subplots(figsize=(10, 6))
    if data is None or len(data) == 0:
        plt.close(fig)
        return
    xs = data[:, 0]
    ys = data[:, 1]
    ax.scatter(xs, ys, c=labels, cmap="tab10", alpha=0.6)
    ax.scatter(xkmeans, ykmeans, marker="X", s=200, c="black", label="centers")
    for i, txt in enumerate(centre_labels):
        ax.annotate(txt, (xkmeans[i], ykmeans[i]), textcoords="offset points", xytext=(5, 5))
    ax.set_title("Clustered data")
    ax.set_xlabel(col1 if 'col1' in locals() else 'Feature 1')
    ax.set_ylabel(col2 if 'col2' in locals() else 'Feature 2')
    ax.legend()
    plt.savefig("clustering.png", bbox_inches="tight")
    plt.close(fig)
    return


def perform_fitting(df, col1, col2):
    # Gather data and prepare for fitting
    if col1 not in df.columns or col2 not in df.columns:
        numcols = df.select_dtypes(include=[np.number]).columns
        if len(numcols) >= 2:
            col1, col2 = numcols[0], numcols[1]
        else:
            raise ValueError("Not enough numeric columns for fitting.")

    dfa = df[[col1, col2]].dropna().astype(float)
    if dfa.shape[0] == 0:
        raise ValueError("No rows with both fitting columns present.")

    x = dfa[col1].values
    y = dfa[col2].values

    # Fit model (simple linear regression)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Predict across x
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    y_line = intercept + slope * x_line

    data = pd.DataFrame({col1: x, col2: y})
    return data, x_line, y_line


def plot_fitted_data(data, x, y):
    fig, ax = plt.subplots(figsize=(10, 6))
    cols = data.columns
    ax.scatter(data[cols[0]], data[cols[1]], alpha=0.6, label="data")
    ax.plot(x, y, linestyle="-", linewidth=2, label="fitted line")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_title(f"Fitting: {cols[1]} vs {cols[0]}")
    ax.legend()
    plt.savefig("fitting.png", bbox_inches="tight")
    plt.close(fig)
    return


def main():
    df = pd.read_csv("data.csv")
    df = preprocessing(df)

    if "price" in df.columns:
    chosen_col = "price"
    else:
    chosen_col = df.select_dtypes(include=[np.number]).columns[0]


    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, chosen_col)
    writing(moments, chosen_col)

    # clustering: pick two features
    cluster_x = "mileage" if "mileage" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    cluster_y = "horsepower" if "horsepower" in df.columns else df.select_dtypes(include=[np.number]).columns[1]
    clustering_results = perform_clustering(df, cluster_x, cluster_y)
    plot_clustered_data(*clustering_results)

    # fitting: milage -> price
    fitting_results = perform_fitting(df, "mileage", "price")
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()
