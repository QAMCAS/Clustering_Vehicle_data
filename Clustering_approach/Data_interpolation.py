import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import json

"""
 Please download the original bus signals data of each city from a2d2: https://www.a2d2.audi/a2d2/en/download.html
 the available locations are "Gaimersheim", "Munich" and "Ingolstadt"
 and save it in the Original_data folder
 
"""

# adapt file path to the Original_data folder.
file_path = 'Original_data/<location>/bus_signals.json'
df = pd.read_json(file_path)
print(df)

acc_x = np.array(df["acceleration_x"]["values"])  # create an array containing the values of acceleration
acc_y = np.array(df["acceleration_y"]["values"])
veh_speed = np.array(df["vehicle_speed"]["values"])

# plotting each sensor separately
fig, axes = plt.subplots(22, 1, figsize=(15, 60))
for (col, ax) in zip(df.columns, axes):
    att = np.array(df[col]["values"])
    ax.plot(att[:, 0], att[:, 1])
    ax.title.set_text(col)
plt.show()

# search for the minimum Timestamp
TS_min = acc_x[:, 0][0]
col_min = "acceleration_x"
print("initial TS", TS_min)
for (col, ax) in zip(df.columns, axes):
    att = np.array(df[col]["values"])
    x = att[:, 0]
    for idx in range(len(x)):
        if x[idx] < TS_min:
            TS_min = x[idx]
            # print(col, ax)
            col_min = col

print("minimal TS ", TS_min, " / ", col_min)

# search for the maximum Timestamp
TS_max = acc_x[:, 0][0]
print("initial TS", TS_max)
for (col, ax) in zip(df.columns, axes):
    att = np.array(df[col]["values"])
    x = att[:, 0]
    for idx in range(len(x)):
        if x[idx] > TS_max:
            TS = x[idx]
            TS_max = TS
            # print(col, ax)
            col_max = col

print("Final TS ", TS, " / ", TS_max, " / ", col_max)

# look for the minimum timestamp difference
min_sensor = np.array(df[col_min]["values"])
min_TS_diff = min_sensor[1][0] - min_sensor[0][0]
Min_column = np.array(df.columns[0])
print(Min_column)
for (col, ax) in zip(df.columns, axes):
    att1 = np.array(df[col]["values"])
    for idx in range(len(att1)):
        x_diff = att1[idx][0] - att1[idx - 1][0]
        if 0 < x_diff < min_TS_diff:
            min_TS_diff = x_diff
            Min_column = col
print("minimium time difference ", min_TS_diff, " ", "Column", Min_column)

# search for the sensor having the min nbr of values
Min_S = np.array(df[df.columns[0]]["values"])
Sensor_MIN = df.columns[0]
for (col, ax) in zip(df.columns, axes):
    S = np.array(df[col]["values"])
    if len(S) < len(Min_S):
        Min_S = S
        Sensor_MIN = col

print("minimum sensor", Min_S, " ", Sensor_MIN, " ", len(Min_S))

print("Here is the size of all sensors values")
for (col, ax) in zip(df.columns, axes):
    print(col, "  ", len(df[col]["values"]))

# #search for the sensor having the max nbr of values
Max_S = np.array(df[df.columns[0]]["values"])
Sensor_MAX = df.columns[0]
print("initial SensorMax", Sensor_MAX)
print(len(df["acceleration_x"]["values"]), " ", len(acc_x[:, 0]))
for (col, ax) in zip(df.columns, axes):
    S = np.array(df[col]["values"])
    if len(S) > len(Max_S):
        print("HEY  ", len(S), len(Min_S))
        Max_S = S
        Sensor_MAX = col

print("maximum sensor", Max_S, " ", Sensor_MAX, " ", len(Max_S))

# create a new timeline
timeline = [TS_min]
i = 0
while timeline[i] < TS_max:
    timeline.append(timeline[i] + min_TS_diff)
    i = i + 1

timeline[len(timeline) - 1] = TS_max

# writing interpolated data  in a json file
new_data_dict = dict()
with open('Interpolated_data/data_Ingolstadt.json', 'w') as f:
    for (col, ax) in zip(df.columns, axes):
        data_list = list()
        att = np.array(df[col]["values"])
        x = att[:, 0]
        y = att[:, 1]
        cs = CubicSpline(x, y)
        y_new = cs(timeline)
        for i in range(len(y_new)):
            data_list.append([int(timeline[i]), y_new[i]])
        new_data_dict[col] = ({'unit': df[col]["unit"], 'values': data_list})

    json.dump(new_data_dict, f, sort_keys=True, indent=4)
