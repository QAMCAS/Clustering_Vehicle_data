import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
  The available locations are "Gaimersheim", "Munich" and "Ingolstadt"
"""

location = 'Gaimersheim'
openfile = 'Interpolated_data/data_' + location + '.json'
df = pd.read_json(openfile)

# Extracting all sensors values
acc_x = np.array(df["acceleration_x"]["values"])
acc_y = np.array(df["acceleration_y"]["values"])
acc_z = np.array(df["acceleration_z"]["values"])

acc_pedal = np.array(df["accelerator_pedal"]["values"])
brake_pressure = np.array(df["brake_pressure"]["values"])
steering_angle_cal = np.array(df["steering_angle_calculated"]["values"])
veh_speed = np.array(df["vehicle_speed"]["values"])

dist_pulse_FL = np.array(df["distance_pulse_front_left"]["values"])
dist_pulse_FR = np.array(df["distance_pulse_front_right"]["values"])
dist_pulse_RearL = np.array(df["distance_pulse_rear_left"]["values"])
dist_pulse_RearR = np.array(df["distance_pulse_rear_right"]["values"])

df_acc = pd.DataFrame(acc_pedal[:, 1])
df_BP = pd.DataFrame(brake_pressure[:, 1])
df_SA = pd.DataFrame(steering_angle_cal[:, 1])
df_VS = pd.DataFrame(veh_speed[:, 1])

df_DPF_L = pd.DataFrame(dist_pulse_FL[:, 1])
df_DPF_R = pd.DataFrame(dist_pulse_FR[:, 1])

acc_pedal_grad = np.array(df["accelerator_pedal_gradient_sign"]["values"])

angular_velocity_omega_x = np.array(df["angular_velocity_omega_x"]["values"])
angular_velocity_omega_y = np.array(df["angular_velocity_omega_y"]["values"])
angular_velocity_omega_z = np.array(df["angular_velocity_omega_z"]["values"])

Latitude_degree = np.array(df["latitude_degree"]["values"])
Latitude_direction = np.array(df["latitude_direction"]["values"])
Longitude_degree = np.array(df["longitude_degree"]["values"])
Longitude_direction = np.array(df["longitude_direction"]["values"])

pitch_angle = np.array(df["pitch_angle"]["values"])
roll_angle = np.array(df["roll_angle"]["values"])

steering_angle_cal = np.array(df["steering_angle_calculated"]["values"])
steering_angle_cal_sign = np.array(df["steering_angle_calculated_sign"]["values"])
veh_speed = np.array(df["vehicle_speed"]["values"])

# plotting distribution histogram of  four main selected sensors values
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
df_acc.plot.hist(grid=True, bins=20, rwidth=0.9, ax=axes[0, 0])
df_BP.plot.hist(grid=True, bins=20, rwidth=0.9, ax=axes[0, 1])
df_SA.plot.hist(grid=True, bins=20, rwidth=0.9, ax=axes[1, 0])
df_VS.plot.hist(grid=True, bins=20, rwidth=0.9, ax=axes[1, 1])
axes[0, 0].title.set_text("Acceleration pedal")
axes[0, 0].set_xlabel("values")
axes[0, 1].title.set_text("Brake pressure ")
axes[0, 1].set_xlabel("values")
axes[1, 0].title.set_text("Steering angle")
axes[1, 0].set_xlabel("values")
axes[1, 1].title.set_text("Vehicle speed")
axes[1, 1].set_xlabel("values")
plt.show()


# Plotting the four main selected sensors values
plt.figure(figsize=(15, 5))
plt.plot(acc_pedal[:, 0], acc_pedal[:, 1])
plt.legend(["Acceleration pedal"])
plt.title("Acceleration pedal")
plt.show()
plt.figure(figsize=(15, 5))
plt.plot(brake_pressure[:, 0], brake_pressure[:, 1])
plt.legend(["Brake pressure"])
plt.title("Brake pressure ")
plt.show()
plt.figure(figsize=(15, 5))
plt.plot(steering_angle_cal[:, 0], steering_angle_cal[:, 1])
plt.legend(["Steering angle"])
plt.title("Steering angle")
plt.show()
plt.figure(figsize=(15, 5))
plt.plot(veh_speed[:, 0], veh_speed[:, 1])
plt.legend(["Vehicle speed"])
plt.title("Vehicle speed")
plt.show()

# changing  data format by creating lists for the arff files
# 2 lists: 1 with specific main four sensors values and one for all sensors
data_list_selectedAtt = []
data_list_allAtt = []

for i in range(len(acc_x[:, 0])):
    data_list_selectedAtt.append(
        [acc_x[i][0], acc_pedal[i][1], brake_pressure[i][1],
         steering_angle_cal[i][1], veh_speed[i][1]])

for i in range(len(acc_x[:, 0])):
    data_list_allAtt.append([acc_x[i][0], acc_x[i][1], acc_y[i][1], acc_z[i][1], acc_pedal[i][1], acc_pedal_grad[i][1],
                             angular_velocity_omega_x[i][1], angular_velocity_omega_y[i][1],
                             angular_velocity_omega_z[i][1],
                             brake_pressure[i][1], dist_pulse_FL[i][1], dist_pulse_FR[i][1], dist_pulse_RearL[i][1],
                             dist_pulse_RearR[i][1],
                             Latitude_degree[i][1], Latitude_direction[i][1], Longitude_degree[i][1],
                             Longitude_direction[i][1],
                             pitch_angle[i][1], roll_angle[i][1], steering_angle_cal[i][1],
                             steering_angle_cal_sign[i][1],
                             veh_speed[i][1]])

data_array = np.array(data_list_selectedAtt)
data_array_all = np.array(data_list_allAtt)

# creating arff file with selected attributes names
arff.dump('arff_data/' + location + '_Selected_att_Clustering_Weka_Inputdata.arff', data_array,
          relation="Audi",
          names=['timestamps',
                 'accelerator_pedal', 'brake_pressure', 'steering_angle_calculated', 'vehicle_speed'])

#  writing manually  an arff file with all sensors
file_arff = open('arff_data/' + location + '_Weka_clustering_allAtt.arff', "w")
delimiter = ","
file_arff.write("@relation AUDI")
file_arff.write("\n")
file_arff.write("Attribute timestamps")
file_arff.write("\n")
for col in df.columns:
    file_arff.write("@Attribute " + str(col) + " real")
    file_arff.write("\n")

file_arff.write("@data")
file_arff.write("\n")
for idx in range(len(data_array_all)):
    row = data_array_all[idx]
    for j in range(len(row)-1):
        file_arff.write(str(row[j]) + delimiter)
    file_arff.write(str(row[len(row)-1]))
    file_arff.write("\n")
file_arff.close()
