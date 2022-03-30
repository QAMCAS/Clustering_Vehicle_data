import traceback
import weka.core.jvm as jvm
import os
import json

from weka.core.dataset import Instances
from weka.clusterers import Clusterer
from weka.clusterers import ClusterEvaluation
from weka.core.converters import Loader
from weka.filters import Filter, MultiFilter
import weka.plot as plot

if plot.matplotlib_available:
    import matplotlib.pyplot as plt
from matplotlib import pyplot
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# logging setup
logger = logging.getLogger(__name__)


def main():
    """
     The available locations are "Gaimersheim", "Munich" and "Ingolstadt"
    """

    location = "Ingolstadt"
    loader = Loader("weka.core.converters.ArffLoader")
    path_to_clean_data = os.path.join("Clustering_input_data/", location, "test_data_cleaned.arff")
    path_to_filtered_data = os.path.join("Clustering_input_data/", location, "test_data_Filtered.arff")
    clean_data = loader.load_file(path_to_clean_data)
    input_data = loader.load_file(path_to_filtered_data)

    List_attributes = []
    for att in range(0, clean_data.num_attributes):
        List_attributes.append(clean_data.attribute(att))
    print(List_attributes)

    List_of_clusterers = ['SimpleKMeans', 'EM', 'Canopy',
                          'SelfOrganizingMap']  # for 'SelfOrganizingMap' you need to remove nb_clusters
    range_n_clusters = [4, 5, 6, 7, 8]

    for nb_clusters in range_n_clusters:

        for algorithm in List_of_clusterers:

            if algorithm != 'SelfOrganizingMap':
                clusterer = Clusterer(classname="weka.clusterers." + str(algorithm), options=["-N", str(nb_clusters)])
                clusterer.build_clusterer(input_data)
                print(clusterer)
                evaluation = ClusterEvaluation()
                evaluation.set_model(clusterer)
                evaluation.test_model(input_data)
                print("eva;" + str(evaluation.cluster_results))
                print("# clusters: " + str(evaluation.num_clusters))
                print("log likelihood: " + str(evaluation.log_likelihood))
                print("cluster assignments:\n" + str(evaluation.cluster_assignments))
                PCORR_file_name = "PCORR_" + str(clusterer.classname) + "_K_" + str(nb_clusters)
                AVG_Percentage_Pearson_correlation(location, input_data, clean_data, clusterer, evaluation,
                                                   PCORR_file_name)

            if algorithm == 'SelfOrganizingMap' and nb_clusters == 4:
                clusterer = Clusterer(classname="weka.clusterers." + str(algorithm))
                clusterer.build_clusterer(input_data)
                print(clusterer)
                evaluation = ClusterEvaluation()
                evaluation.set_model(clusterer)
                evaluation.test_model(input_data)
                print("eva;" + str(evaluation.cluster_results))
                print("# clusters: " + str(evaluation.num_clusters))
                print("log likelihood: " + str(evaluation.log_likelihood))
                print("cluster assignments:\n" + str(evaluation.cluster_assignments))
                PCORR_file_name = "PCORR_" + str(clusterer.classname) + "_K_" + str(nb_clusters)
                AVG_Percentage_Pearson_correlation(location, input_data, clean_data, clusterer, evaluation,
                                                   PCORR_file_name)

        # Using autoencoder
        print("**** Clustering using auto-encoder ****")
        Autoencoder = Filter(classname="weka.filters.unsupervised.attribute.MLPAutoencoder")
        Autoencoder.inputformat(input_data)
        filtered = Autoencoder.filter(input_data)  # data filtered with  autoencoder
        clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", str(nb_clusters)])
        clusterer.build_clusterer(filtered)
        print(clusterer)
        evaluation = ClusterEvaluation()
        evaluation.set_model(clusterer)
        evaluation.test_model(filtered)

        PCORR_file_name = "PCORR_" + str(clusterer.classname) + "withAutoencoder_K_" + str(nb_clusters)
        AVG_Percentage_Pearson_correlation(location, input_data, clean_data, clusterer, evaluation, PCORR_file_name)


# write extracted driving scenarios in json files
def create_JSON_files(location, data_filtered, clusterer, evaluation):
    cluster_data_path = 'Results/driving_scenarios/'
    for cluster in range(clusterer.number_of_clusters):

        with open(os.path.join(cluster_data_path, location, 'json_files',
                               str(clusterer.classname) + '_scenario_' + str(cluster) + '.json'),
                  'w') as f:

            episodes_list = []
            for att in data_filtered.attribute_names():
                val_dict[att] = {'values': []}

            for inst in range(0, len(evaluation.cluster_assignments) - 1):
                if evaluation.cluster_assignments[inst] == cluster:
                    for i in range(len(data_filtered.attribute_names())):
                        att_name = data_filtered.attribute_names()[i]
                        inst_att_value = data_filtered.get_instance(inst).get_value(i)
                        val_dict[att_name]['values'].append(inst_att_value)
                    if evaluation.cluster_assignments[inst + 1] != cluster and len(
                            val_dict['timestamps']['values']) >= 150:
                        episodes_list.append(val_dict)
                        val_dict = {}
                        for att in data_filtered.attribute_names():
                            val_dict[att] = {'values': []}

            ep_dict = {'Episodes': []}
            for j in range(0, len(episodes_list)):
                ep_dict['Episodes'].append({'id_{}'.format(j): {'Sensors': episodes_list[j]}})

            json.dump(ep_dict, f, sort_keys=True, indent=4)

    print("Writing json files  for each cluster is done")


# compute percentage of Pearson correlation in each cluster save PCorr results in a txt file
def AVG_Percentage_Pearson_correlation(location, data_filtered, clean_data, clusterer, evaluation, PCORR_file_name):
    file_correlation = open(
        "Results/" + location + "/" + PCORR_file_name + ".txt",
        "w")
    delimiter = ";"
    file_correlation.write(
        "Cluster" + delimiter + "Nb of total episodes" + delimiter + "Percentage of Pearson correlation")
    file_correlation.write("\n")
    sum_pcorr_cluster = 0
    for cluster in range(0, clusterer.number_of_clusters):
        fig = plt.figure(figsize=(8, 6))
        Sum_pcorr = 0
        file_correlation.write(str(cluster) + delimiter)
        for (att, num) in zip(range(0, data_filtered.num_attributes), range(1, 5)):
            List_episodes = []
            List_timestamps = []
            val_list = []
            for inst in range(len(evaluation.cluster_assignments) - 1):
                if evaluation.cluster_assignments[inst] == cluster:
                    # print("ok3")
                    val_list.append(data_filtered.get_instance(inst).get_value(att))
                    List_timestamps.append(clean_data.get_instance(inst).get_value(att))
                    if evaluation.cluster_assignments[inst + 1] != cluster and len(val_list) > 20:
                        List_episodes.append(val_list)
                        val_list = []
            print(" number of episodes in cluster ", cluster, "for ", str(data_filtered.attribute(att)), " ",
                  len(List_episodes))

            if len(List_episodes) > 0:
                df = pd.DataFrame(List_episodes)
                df_t = df.T
                df_t.dropna()
                ax = fig.add_subplot(2, 2, num)
                fig.suptitle(clusterer.classname + "/Cluster " + str(cluster))
                ax.set_title(str(data_filtered.attribute_names()[att]))
                ax.plot(df_t)

                corr = df_t.corr(method='pearson', min_periods=1)
                List_i = []
                for i in range(0, len(corr)):
                    nbr_correlated_ep_perLine = 0
                    for j in range(0, len(corr[i])):
                        if i < j:
                            if corr[i][j] >= 0.5:
                                if j not in List_i:
                                    List_i.append(j)
                                    nbr_correlated_ep_perLine += 1

                total_correlated_ep_per_att = len(List_i)
                print("total of correlated episodes per attribute", total_correlated_ep_per_att)
                pcorr_per_att = (total_correlated_ep_per_att / len(List_episodes)) * 100
                pcorr_per_att = round(pcorr_per_att, 2)
                print("percentage of  pearson correlation_per_att", pcorr_per_att)
                Sum_pcorr = Sum_pcorr + pcorr_per_att

        avg_pcorr_cluster = round(Sum_pcorr / data_filtered.num_attributes, 2)
        sum_pcorr_cluster = sum_pcorr_cluster + avg_pcorr_cluster
        file_correlation.write(
            str(len(List_episodes)) + delimiter + str(avg_pcorr_cluster))
        file_correlation.write("\n")
        plt.tight_layout
        plt.show()
    avg_pcorr_final = round(sum_pcorr_cluster / clusterer.number_of_clusters, 2)

    print("Episodes evaluation is done")
    file_correlation.write("\n")
    file_correlation.write(str(avg_pcorr_final))
    file_correlation.close()


# extract episodes
def ExtractEpisodes(data_filtered, cluster, evaluation, att):
    List_episodes = []
    val_list = []
    for inst in range(len(evaluation.cluster_assignments) - 1):
        if evaluation.cluster_assignments[inst] == cluster:
            val_list.append(data_filtered.get_instance(inst).get_value(att))
            if evaluation.cluster_assignments[inst + 1] != cluster and len(val_list) > 20:
                List_episodes.append(val_list)
                val_list = []
    return List_episodes


# plots graphs of extracted episodes, probability distribution and Pearson correlation matrix for each attribute and
# save their files
def Plot_Episodes_ProbabilityDist_PearsonCorrMatrix(location, input_data, clean_data, clusterer, evaluation):
    for cluster in range(0, clusterer.number_of_clusters):
        # np.os.mkdir("Results/" + location + "/Cluster_" + str(cluster))
        path_to_file = "Results/" + location + "/Cluster_" + str(cluster) + "/"

        for (att, num) in zip(range(0, input_data.num_attributes), range(1, 5)):
            List_episodes = []
            List_timestamps = []
            val_list = []
            for inst in range(len(evaluation.cluster_assignments) - 1):
                if evaluation.cluster_assignments[inst] == cluster:
                    # print("ok3")
                    val_list.append(input_data.get_instance(inst).get_value(att))
                    List_timestamps.append(clean_data.get_instance(inst).get_value(att))
                    if evaluation.cluster_assignments[inst + 1] != cluster and len(val_list) > 20:
                        List_episodes.append(val_list)
                        val_list = []

            if len(List_episodes) > 0:
                df = pd.DataFrame(List_episodes)
                df_t = df.T
                df_t.dropna()

                # Plot extracted episodes for each attribute
                plt.figure(figsize=(10, 20))
                df_t.plot(legend=True)
                plt.title("Episodes " + str(input_data.attribute_names()[att]) + " in cluster " + str(cluster))
                plt.savefig(path_to_file + "Episodes " + str(input_data.attribute(att)) + "_cluster_" + str(cluster))
                plt.show()

                # Plot episodes probability distribution  for each attribute
                df_t.plot.hist(density=True, legend=False, grid=False, bins=20, rwidth=0.9)
                plt.xlabel('Values')
                plt.ylabel('Probability density distribution')
                plt.grid(axis='x', alpha=0.75)
                plt.title(str(input_data.attribute_names()[att]) + " in cluster  " + str(cluster))
                plt.savefig(
                    path_to_file + "Probability_Distribution_" + str(input_data.attribute(att)) + "_cluster_" + str(
                        cluster))
                plt.show()

                # Plot Pearson coefficient matrix for each attribute
                corr = df_t.corr(method='pearson', min_periods=1)
                mask = np.zeros_like(corr, dtype='bool')
                f, ax = pyplot.subplots(figsize=(12, 10))
                cmap = sns.diverging_palette(220, 10, as_cmap=True)
                sns.heatmap(corr, mask=mask, cmap=cmap,
                            square=True, annot=True, linewidths=.5, ax=ax)

                pyplot.title("Pearson Correlation matrix of the episodes  " + str(
                    input_data.attribute_names()[att]) + " in cluster  " + str(cluster))
                plt.savefig(path_to_file + "Pearson Correlation matrix of the episodes_" + str(
                    input_data.attribute(att)) + "_cluster_" + str(cluster))


def Create_arff_outputfiles(data_cleaned, location, clusterer, evaluation, List_attributes):
    # Writing extracted driving scenarios  in  arff file for weka use
    for cluster in range(0, clusterer.number_of_clusters):
        dataset_scenario = Instances.create_instances("scenario_" + str(cluster), List_attributes, 0)
        for att in range(0, data_cleaned.num_attributes):
            timestamps_list = []
            val_list = []
            stop = False
            for inst in range(len(evaluation.cluster_assignments)):
                if evaluation.cluster_assignments[inst] == cluster and stop == False:
                    val_list.append(data_cleaned.get_instance(inst).get_value(att))
                    timestamps_list.append(data_cleaned.get_instance(inst).get_value(0))
                    dataset_scenario.add_instance(data_cleaned.get_instance(inst))

                    if evaluation.cluster_assignments[inst + 1] != cluster and len(val_list) > 20:
                        stop = True

        file_scenario = open(
            "Results/driving_scenarios/" + location + "/arff_files/" + str(clusterer.classname) + "_Scenario_" + str(
                cluster) + ".arff", "w")
        file_scenario.write(str(dataset_scenario))

    file_scenario = open(
        "Results/dataset_scenarios/" + location + "/arff_files/" + str(clusterer.classname) + "_Scenario_" + str(
            cluster) + ".arff", "w")
    file_scenario.write(str(dataset_scenario))

    file_scenario.close()
    print("Writing results in txt file and arff file  is done")


if __name__ == "__main__":
    try:
        jvm.start(system_cp=True, packages=True, max_heap_size="512m")
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
