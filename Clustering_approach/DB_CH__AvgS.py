# Using Weka clusterer - requires python-weka-wrapper3 later than 0.2.4
import sklweka.jvm as jvm
from sklweka.clusters import WekaCluster
from sklweka.dataset import load_arff
import pandas as pd
from weka.core.converters import Loader
from weka.filters import Filter
import traceback
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def main():
    loader = Loader("weka.core.converters.ArffLoader")
    """
    The available locations are "Gaimersheim", "Munich" and "Ingolstadt"
    """

    location = "Gaimersheim"
    filename = "Clustering_input_data/"+location+"/test_data_Filtered.arff"
    List_of_clusterers = ['SelfOrganizingMap', 'SimpleKMeans', 'EM', 'Canopy']
    cluster_data, _ = load_arff(filename, class_index=None)
    data = loader.load_file(filename)
    df = pd.DataFrame(data, columns=data.attribute_names())
    compareClusterers(data, df, cluster_data)
    Autoencoder_activated = "autoencoder"  # activate when using auto-encoder
    compare_nb_clusters(Autoencoder_activated, data, df, cluster_data)


def compare_nb_clusters(Autoencoder_activated, data, df, cluster_data):
    range_n_clusters = [4, 5, 6, 7, 8]
    list_DB_index = []
    list_CH_index = []
    for n_clusters in range_n_clusters:
        if Autoencoder_activated == "autoencoder":
            Autoencoder = Filter(classname="weka.filters.unsupervised.attribute.MLPAutoencoder")
            Autoencoder.inputformat(data)
            filtered = Autoencoder.filter(data)
            df_filtered = filtered.to_numpy()  # use for clustering
            df1 = pd.DataFrame(filtered, columns=filtered.attribute_names())  # use for scores
            print(df1.shape)
            clusterer = WekaCluster(classname="weka.clusterers.SimpleKMeans", options=["-N", str(n_clusters)])
            cluster_labels = clusterer.fit_predict(df_filtered)
            DB_index = davies_bouldin_score(df1, cluster_labels)
            CH_index = metrics.calinski_harabasz_score(df1, cluster_labels)
        else:
            clusterer = WekaCluster(classname="weka.clusterers.SimpleKMeans", options=["-N", str(n_clusters)])
            cluster_labels = clusterer.fit_predict(cluster_data)
            DB_index = davies_bouldin_score(df, cluster_labels)
            CH_index = metrics.calinski_harabasz_score(df, cluster_labels)

        print(
            "For algorithm " + str(clusterer.classname), " using  n_clusters =", n_clusters, " DB index = ", DB_index,
            "and CH index =", CH_index)
        list_DB_index.append(DB_index)
        list_CH_index.append(CH_index)

    plt.figure(figsize=(12, 8))
    plt.plot(range_n_clusters, list_DB_index)
    plt.xlabel('number of clusters')
    plt.ylabel('DB Index')
    plt.title('The DB index plot for different clusters for ' + str(clusterer.classname))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(range_n_clusters, list_CH_index)
    plt.xlabel('number of clusters')
    plt.ylabel('CH Index')
    plt.title('The CH index plot for different clusters for ' + str(clusterer.classname))
    plt.show()


def compareClusterers(data, df, cluster_data):
    List_of_clusterer = ['SimpleKMeans', 'EM', 'Canopy',
                         'SelfOrganizingMap']
    range_n_clusters = [4, 5, 6, 7, 8]
    for nclusters in range_n_clusters:
        print("Results for k = ", nclusters)
        for algorithm in List_of_clusterer:
            if algorithm != 'SelfOrganizingMap':
                clusterer = WekaCluster(classname="weka.clusterers." + str(algorithm), options=["-N", str(nclusters)])
                cluster_labels = clusterer.fit_predict(cluster_data)
                DB_index = davies_bouldin_score(df, cluster_labels)
                CH_index = metrics.calinski_harabasz_score(df, cluster_labels)
                silhouette_avg = silhouette_score(df, cluster_labels)
                print(
                    "For algorithm " + str(clusterer.classname), " DB index = ", round(DB_index, 4), "and CH index =",
                    int(CH_index), "AVG silhouette_score  = ", round(silhouette_avg, 4))

            if algorithm == 'SelfOrganizingMap' and nclusters == 4:
                clusterer = WekaCluster(classname="weka.clusterers." + str(algorithm))
                cluster_labels = clusterer.fit_predict(cluster_data)
                DB_index = davies_bouldin_score(df, cluster_labels)
                CH_index = metrics.calinski_harabasz_score(df, cluster_labels)
                silhouette_avg = silhouette_score(df, cluster_labels)
                print(
                    "For algorithm " + str(clusterer.classname), " DB index = ", round(DB_index, 4), "and CH index =",
                    int(CH_index), "AVG silhouette_score  = ", round(silhouette_avg, 4))

        Autoencoder = Filter(classname="weka.filters.unsupervised.attribute.MLPAutoencoder")
        Autoencoder.inputformat(data)
        filtered = Autoencoder.filter(data)
        df_filtered = filtered.to_numpy()  # use for clustering
        df1 = pd.DataFrame(filtered, columns=filtered.attribute_names())  # use for scores
        print(df1.shape)
        clusterer = WekaCluster(classname="weka.clusterers.SimpleKMeans", options=["-N", str(nclusters)])
        cluster_labels = clusterer.fit_predict(df_filtered)
        DB_index = davies_bouldin_score(df1, cluster_labels)
        CH_index = metrics.calinski_harabasz_score(df1, cluster_labels)
        silhouette_avg = silhouette_score(df1, cluster_labels)
        print(
            "For algorithm " + str(clusterer.classname), "with MLPAutoencoder ", " DB index = ", round(DB_index, 4),
            "and CH index =", int(CH_index), "AVG silhouette_score  = ", round(silhouette_avg, 4))


if __name__ == "__main__":
    try:
        jvm.start(system_cp=True, packages=True, max_heap_size="512m")
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
