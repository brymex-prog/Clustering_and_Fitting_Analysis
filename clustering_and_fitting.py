"""
Author: Ibrahim Olayinka Abdulsalam
Clustering and Fitting Project 
Completed code using the Used Car Price Prediction Dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


def plot_relational_plot(df):
    """Create and save a scatter plot of mileage vs price."""
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

    sns.scatterplot(
        data=data, x=xcol, y=ycol, ax=ax, alpha=0.7
    )
    ax.set_title(f"{ycol} vs {xcol}")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)

    plt.tight_layout()
    plt.savefig("relational_plot.png")
    plt.close(fig)
    return


def plot_categorical_plot(df):
    """Create and save a bar plot of average price by fuel type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if "fuel_type" in df.columns and "price" in df.columns:
        agg = (
            df.groupby("fuel_type")["price"]
            .mean()
            .sort_values(ascending=False)
        )

        if agg.shape[0] == 0:
            ax.text(0.5, 0.5, "No fuel_type data", ha="center")
        else:
            sns.barplot(x=agg.index, y=agg.values, ax=ax)
            plt.setp(ax.get_xticklabels(), rotation=0)

        ax.set_title("Average price by fuel type")
        ax.set_xlabel("Fuel type")
        ax.set_ylabel("Average price")
    else:
        ax.text(
            0.5, 0.5,
            "fuel_type or price missing",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig("categorical_plot.png")
    plt.close(fig)
    return


def plot_statistical_plot(df):
    """Create and save a correlation heatmap."""
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 6))

    if numeric.shape[1] >= 2:
        sns.heatmap(
            numeric.corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Correlation Heatmap")
    else:
        ax.text(0.5, 0.5, "Not enough numeric columns", ha="center")

    plt.tight_layout()
    plt.savefig("statistical_plot.png")
    plt.close(fig)


def statistical_analysis(df, col):
    """Return mean, stddev, skewness, and kurtosis."""
    if col not in df.columns:
        numcols = df.select_dtypes(include=[np.number]).columns
        if len(numcols) == 0:
            raise ValueError(
                "No numeric columns available for analysis."
            )
        col = numcols[0]

    series = df[col].dropna().astype(float)
    mean = float(series.mean())
    stddev = float(series.std(ddof=0))
    skew = float(ss.skew(series))
    kurt = float(ss.kurtosis(series, fisher=True))

    return mean, stddev, skew, kurt


def preprocessing(df):
    """Clean raw dataset and extract numeric fields."""
    df = df.copy()

    colmap = {}
    if "price" in df.columns:
        colmap["price"] = "price"
    if "milage" in df.columns:
        colmap["milage"] = "mileage"
    if "model_year" in df.columns:
        colmap["model_year"] = "year"
    if "engine" in df.columns:
        colmap["engine"] = "engine"

    df = df.rename(columns=colmap)

    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("$", "")
            .str.replace(",", "")
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "mileage" in df.columns:
        df["mileage"] = df["mileage"].astype(str).str.replace(",", "")
        df["mileage"] = df["mileage"].str.extract(
            r"(\d+\.?\d*)"
        ).astype(float)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["age"] = 2025 - df["year"]

    if "engine" in df.columns:
        df["engine_cc"] = (
            df["engine"]
            .astype(str)
            .str.extract(r"(\d+\.?\d*)")
            .astype(float)
        )

    df = df.dropna(subset=["price", "mileage", "age"])
    return df


def writing(moments, col):
    """Print moment statistics."""
    print(f"For the attribute {col}:")
    print(
        f"Mean = {moments[0]:.2f}, "
        f"Standard Deviation = {moments[1]:.2f}, "
        f"Skewness = {moments[2]:.2f}, "
        f"Excess Kurtosis = {moments[3]:.2f}."
    )


def perform_clustering(df, col1, col2):
    """Perform KMeans clustering on two variables."""

    def plot_elbow(inertias):
        fig, ax = plt.subplots(figsize=(10, 6))
        ks = list(range(2, 2 + len(inertias)))
        ax.plot(ks, inertias, marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Plot")
        plt.savefig("elbow_plot.png")
        plt.close(fig)

    if col1 not in df.columns or col2 not in df.columns:
        num = df.select_dtypes(include=[np.number]).columns
        col1, col2 = num[0], num[1]

    data_df = df[[col1, col2]].dropna().astype(float)
    data = data_df.values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    inertias = []
    best_k = 2
    best_score = -1

    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        inertias.append(km.inertia_)

        try:
            score = silhouette_score(scaled, labels)
        except Exception:
            score = -1

        if score > best_score:
            best_k = k
            best_score = score

    plot_elbow(inertias)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    centers = scaler.inverse_transform(km.cluster_centers_)

    xk = centers[:, 0]
    yk = centers[:, 1]
    clabels = [f"c{i}" for i in range(len(xk))]

    return labels, data, xk, yk, clabels, col1, col2


def plot_clustered_data(labels, data, xk, yk, clabels, col1, col2):
    """Plot clustered result."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        data[:, 0], data[:, 1],
        c=labels, cmap="tab10", alpha=0.6
    )
    ax.scatter(
        xk, yk, marker="X",
        s=200, c="black", label="centers"
    )

    for i, txt in enumerate(clabels):
        ax.annotate(
            txt, (xk[i], yk[i]),
            textcoords="offset points",
            xytext=(5, 5)
        )

    ax.set_title("Clustered Data")
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.legend()

    plt.savefig("clustering.png", bbox_inches="tight")
    plt.close(fig)


def perform_fitting(df, col1, col2):
    """Perform simple linear regression fitting."""
    data = df[[col1, col2]].dropna().astype(float)
    x = data[col1].values
    y = data[col2].values

    slope, intercept, _, _, _ = stats.linregress(x, y)

    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = intercept + slope * x_line

    return data, x_line, y_line


def plot_fitted_data(data, x, y):
    """Plot data and fitted line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cols = data.columns
    ax.scatter(
        data[cols[0]], data[cols[1]],
        alpha=0.6, label="data"
    )
    ax.plot(x, y, linewidth=2, label="fitted line", color='red')

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_title(f"Fitting: {cols[1]} vs {cols[0]}")
    ax.legend()

    plt.savefig("fitting.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv("data.csv")
    df = preprocessing(df)

    # Quick inspection features
    print("\n===== FIRST 5 ROWS (head) =====")
    print(df.head())

    print("\n===== LAST 5 ROWS (tail) =====")
    print(df.tail())

    print("\n===== SUMMARY STATISTICS (describe) =====")
    print(df.describe(include="all"))

    print("\n===== CORRELATION MATRIX (corr) =====")
    print(df.corr(numeric_only=True))

    if "price" in df.columns:
        chosen_col = "price"
    else:
        chosen_col = (
            df.select_dtypes(include=[np.number]).columns[0]
        )

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, chosen_col)
    writing(moments, chosen_col)

    if "mileage" in df.columns:
        cluster_x = "mileage"
    else:
        cluster_x = (
            df.select_dtypes(include=[np.number]).columns[0]
        )

    if "horsepower" in df.columns:
        cluster_y = "horsepower"
    else:
        cluster_y = (
            df.select_dtypes(include=[np.number]).columns[1]
        )

    results = perform_clustering(df, cluster_x, cluster_y)
    plot_clustered_data(*results)

    fit = perform_fitting(df, "mileage", "price")
    plot_fitted_data(*fit)


if __name__ == "__main__":
    main()
