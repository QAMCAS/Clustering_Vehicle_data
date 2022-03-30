# Clustering_Vehicle_data

This repository includes modified data and source code used in the paper "On the application of clustering for extracting driving scenarios from vehicle data
",  submitted and under review to the journal "Machine Learning with Applications"

# Paper summary
If we want to extract test cases from driving data with the purpose of testing vehicles, we want to avoid using similar test cases. In this paper, we
focus on this topic. We provide a method for extracting driving episodes from data utilizing clustering algorithms. This method starts with clustering driving data. Afterward, data points representing time-ordered sequences
are obtained from the cluster forming a driving episode. Besides outlying the foundations, we present the results of an experimental evaluation where
we considered six different clustering algorithms and available driving data from three German cities. To evaluate the cluster quality, we utilize three cluster validity metrics. In addition, we introduce a measure for the quality
of extracted episodes relying on the Pearson coefficient. The experimental evaluation shows that the cluster validity metrics do not provide good results. The Pearson coefficient allows ranking the clustering algorithms. The
carried out experimental evaluation leads to the following results. We can extract meaningful episodes from driving data using any clustering algorithm considering four to eight clusters. Combining k-means clustering with auto-
encoders leads to the best Pearson correlation. SOM is the slowest clustering method, and Canopy is the fastest.

# Original data:
In this work we make use of the public available A2D2 data [1]  originally downloaded from https://www.a2d2.audi/a2d2/en/download.html

For the clustering approach we used the Bus Signals data available for three German cities: Gaimersheim, Munich and Ingolstadt. 

For the driving scenarios validation we make use of the Camera-FrontCenter images.

The original data is not provided in this repository, in order to download the initial bus signals data and camera images, please refer to above-mentioned link.
For the purpose of our study we performed some changes on the original data, here we only provide the modified data used in each step of our approach.  

[1] Geyer, Jakob, et al. "A2d2: Audi autonomous driving dataset." arXiv preprint arXiv:2004.06320 (2020).
# Necessary Libraries:
in order to be able to run all algorithms available in the Python wrapper for the Java machine learning workbench Weka (see: https://www.cs.waikato.ac.nz/~ml/weka/), it is mandatory to install these libraries:  
 
1. python-weka-wrapper3 : https://github.com/fracpete/python-weka-wrapper3

2. python-javabridge 4.0.3 : https://pypi.org/project/python-javabridge/

3. sklearn-weka-plugin  which makes Weka algorithms available in scikit-learn. https://github.com/fracpete/sklearn-weka-plugin



# Repository structure
The project is composed of two main folders:

I/ Clustering approach:

   1. Data_interpolation: This script performs Cubic spline interpolation on the original data in order to synchronize all bus signals values. The intperolated data for each city  is saved in the  "Interpolated_data" folder.

   2. Convert_Interpolated_data: This script converts the interpolated data to the .arff format which is the required format used in WEKA.
   In this work we only make use of four main sensors which are : 'accelerator_pedal', 'brake_pressure', 'steering_angle_calculated', 'vehicle_speed' but it is possible to use all sensors if required. The data in .arff format for each city is saved in the "arff_data" folder.  
   3. Data_cleansing: This script replaces all brake pressure values which are <= 0.2 to 0 in order to obtain  more precise results. Also it filters out the Timestamps attribute values as it is not considered when clustering the data. The clean data (including timestamps) and filtered data (without timestamps) is saved in "Clustering-input_data" folder"
   4. Data_clustering: In this script, we perform clustering using five different algorithms provided by weka (Simple k-means, EM, Canopy clustering, SOM clustering and K-means using Auto-encoder pre-processing ). We save the extracted driving scenarios in .json files using the function: create_JSON_files, plus we provide the option of saving driving scenarios in .arff format if needed to be used as input to weka. For this we use the function:Create_arff_outputfiles. The extracted driving scenarios are saved in both formats in  Results/driving_scenarios folder.  Also, in this script we compute the average percentage of Pearson correlation in each cluster using the function AVG_Percentage_Pearson_correlation. The Pearson correlation results  for each city are saved in .txt format in the Results folder. We also provide some other graphs of extracted episodes, their probability distribution and Pearson correlation matrices for each attribute obtained in a previous study (see: https://web.archive.org/web/20211013034723id_/http://ksiresearch.org/seke/seke21paper/paper118.pdf)    
   5. DB_CH_AvgS:  computes The Davies-Bouldin index (DB), The Calinski-Harabasz index (CH) and The Silhouette index (S) for each algorithm using different numbers of clusters. These three metrics are available in Scikit-learn library. 
   6. Plot_silhouette: Plot the silhouette results for different numbers of clusters.


II/ Camera_images_Validation:

In this folder, we provide scripts for matching camera images available in  the original A2D2 data to extracted driving scenarios in obtained clusters. 
 1. img2vid.py: This script creates videos for each city using the original Camera Front_center images downloaded from the link above.  
 2. synchro_vid_clustered_bus: Synchronizes the clustered bus signals with camera images and matches each driving scenario (available in Clustering_approach/Results/driving_scenarios) to corresponding sequences of images.
    

