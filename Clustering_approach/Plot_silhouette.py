# Using Weka clusterer - requires python-weka-wrapper3 later than 0.2.4
# This script is based on the silhouette example from scikit-learn
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

from sklearn.metrics import silhouette_samples, silhouette_score
import sklweka.jvm as jvm
from sklweka.clusters import WekaCluster
from sklweka.dataset import load_arff
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from weka.core.converters import Loader
from weka.filters import Filter

jvm.start(packages=True)
loader = Loader("weka.core.converters.ArffLoader")
"""
The available locations are "Gaimersheim", "Munich" and "Ingolstadt"
"""
location = "Gaimersheim"
clean_data = loader.load_file("Clustering_input_data/" + location + "/test_data_cleaned.arff")
fname = "Clustering_input_data/" + location + "/test_data_Filtered.arff"

df_clean = pd.DataFrame(clean_data, columns=clean_data.attribute_names())
data = loader.load_file(fname)
X, y, _ = load_arff(fname, class_index='last')
cluster_data, _ = load_arff(fname, class_index=None)

# outputting the shape of the dataset
df = pd.DataFrame(data, columns=data.attribute_names())

Autoencoder = Filter(classname="weka.filters.unsupervised.attribute.MLPAutoencoder")
Autoencoder.inputformat(data)
filtered = Autoencoder.filter(data)
df_filtered = filtered.to_numpy()  # use for clustering with auto-encoder
df_AE = pd.DataFrame(filtered, columns=filtered.attribute_names())  # use for silhouette with auto-encoder

range_n_clusters = [4, 5, 6, 7, 8]
for n_clusters in range_n_clusters:

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    List_of_clusterers = ['SimpleKMeans', 'EM', 'Canopy', 'SelfOrganizingMap']
    clusterer = WekaCluster(classname="weka.clusterers.EM", options=["-N", str(n_clusters)])
    cluster_labels = clusterer.fit_predict(cluster_data)  # change to df_filtered when using auto-encoder

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(df, cluster_labels)  # change to df_AE when using auto-encoder
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        df_clean['timestamps'], df['vehicle_speed'], marker=".", s=30, lw=0, alpha=0.7, c=colors,
        edgecolor="k"
    )

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for " + str(clusterer.classname)
        + " on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()

jvm.stop()
