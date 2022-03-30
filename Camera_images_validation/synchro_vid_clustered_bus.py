import glob
import matplotlib.pyplot as plt
import json
import os
import cv2 as cv
import numpy as np


def cv_frame_count(cap_or_path):
    had_to_create_video_capture = False
    cap = cap_or_path  # cap_or_path is used as the argument to make clear what is accepted as input

    if isinstance(cap, str):
        had_to_create_video_capture = True
        assert os.path.isfile(cap)
        cap = cv.VideoCapture(cap)
        assert cap.isOpened()

    assert type(cap).__module__ == 'cv2'
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    assert frame_count.is_integer()

    if had_to_create_video_capture:
        cap.release()

    return int(frame_count)


# return the index of the current frame (first frame is index 0)
def cv_current_frame(cap):
    x = cap.get(cv.CAP_PROP_POS_FRAMES)
    assert x.is_integer()
    return int(x)


# jump to frame index 'frame_id'
def cv_goto_frame(cap, frame_id):
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
    assert cv_current_frame(cap) == frame_id


# remove the extension from the path and return
def without_ext(path):
    return os.path.splitext(path)[0]


def play_vid(vid_path):
    if isinstance(vid_path, str):
        assert os.path.isfile(vid_path)
        cap = cv.VideoCapture(vid_path)
        assert cap.isOpened()

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            cv.imshow("Frame", frame)

            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release

    cv.destroyAllWindows()


def load_bus_data(bus_data_path):
    bus_data_dict = {}
    with open(bus_data_path) as bus_f:
        bus_data = json.load(bus_f)

    for key in bus_data.keys():
        bus_data_dict[key] = {}
        bus_data_dict[key]['unit'] = bus_data[key]["unit"]
        bus_data_dict[key]["timestamps"] = []
        bus_data_dict[key]["values"] = []
        for value in bus_data[key]["values"]:
            bus_data_dict[key]["timestamps"].append(value[0])
            bus_data_dict[key]["values"].append(value[1])

    # bus signal - [unit]
    # acceleration_x - MeterPerSeconSquar
    # acceleration_y - MeterPerSeconSquar
    # acceleration_z - MeterPerSeconSquar
    # accelerator_pedal  - PerCent
    # accelerator_pedal_gradient_sign - null
    # angular_velocity_omega_x - DegreOfArcPerSecon
    # angular_velocity_omega_y - DegreOfArcPerSecon
    # angular_velocity_omega_z - DegreOfArcPerSecon
    # brake_pressure  - Bar
    # distance_pulse_front_left - null
    # distance_pulse_front_right - null
    # distance_pulse_rear_left - null
    # distance_pulse_rear_right  - null
    # latitude_degree  - DegreOfArc
    # latitude_direction - null
    # longitude_degree  - DegreOfArc
    # longitude_direction  - null
    # pitch_angle  - DegreOfArc
    # roll_angle  - DegreOfArc
    # steering_angle_calculated  - DegreOfArc
    # steering_angle_calculated_sign  - null
    # vehicle_speed  - KiloMeterPerHour

    return bus_data_dict


def load_cluster_scenario_data(bus_data_path):
    bus_data_dict = {}
    with open(bus_data_path) as bus_f:
        bus_data = json.load(bus_f)

    for ep_id in bus_data['Episodes']:
        for key in ep_id.keys():
            bus_data_dict[key] = {}
            for sensor in ep_id[key]['Sensors']:
                if sensor != 'timestamps':
                    bus_data_dict[key][sensor] = {}
                    bus_data_dict[key][sensor]['timestamps'] = ep_id[key]['Sensors']['timestamps']['values']
                    bus_data_dict[key][sensor]['values'] = ep_id[key]['Sensors'][sensor]['values']

    # bus signal - [unit]
    # acceleration_x - MeterPerSeconSquar
    # acceleration_y - MeterPerSeconSquar
    # acceleration_z - MeterPerSeconSquar
    # accelerator_pedal  - PerCent
    # accelerator_pedal_gradient_sign - null
    # angular_velocity_omega_x - DegreOfArcPerSecon
    # angular_velocity_omega_y - DegreOfArcPerSecon
    # angular_velocity_omega_z - DegreOfArcPerSecon
    # brake_pressure  - Bar
    # distance_pulse_front_left - null
    # distance_pulse_front_right - null
    # distance_pulse_rear_left - null
    # distance_pulse_rear_right  - null
    # latitude_degree  - DegreOfArc
    # latitude_direction - null
    # longitude_degree  - DegreOfArc
    # longitude_direction  - null
    # pitch_angle  - DegreOfArc
    # roll_angle  - DegreOfArc
    # steering_angle_calculated  - DegreOfArc
    # steering_angle_calculated_sign  - null
    # vehicle_speed  - KiloMeterPerHour

    return bus_data_dict


def extract_image_timestamps(cam_images_files_json):
    img_timestamps = []
    for image_path in cam_images_files_json:
        with open(image_path) as image_json_f:
            image_data = json.load(image_json_f)
            img_timestamps.append(image_data["cam_tstamp"])

    return img_timestamps


def get_closest_value_idx(bus_data, img_tstamps):
    k_min = img_tstamps[0]
    k_max = img_tstamps[-1]
    lst = np.asarray(bus_data)
    idx_min = (np.abs(lst - k_min)).argmin()
    idx_max = (np.abs(lst - k_max)).argmin()

    return idx_min, idx_max


def get_closest_value_clustered_idx(data_tstamps, img_tstamps):
    k_min = data_tstamps[0]
    k_max = data_tstamps[-1]
    lst = np.asarray(img_tstamps)
    idx_min = (np.abs(lst - k_min)).argmin()
    idx_max = (np.abs(lst - k_max)).argmin()

    return idx_min, idx_max


def interpolate_bus_data(bus_data, img_tstamps, data_o_inter):
    y_interp_array = []

    for data in data_o_inter:
        data_tstamps = bus_data[data]["timestamps"]
        data_values = bus_data[data]["values"]

        idx_min, idx_max = get_closest_value_idx(data_tstamps, img_tstamps)

        data_tstamps_refactored = data_tstamps[idx_min:idx_max]
        data_values_refactored = data_values[idx_min:idx_max]

        xvals = img_tstamps
        yinterp = np.interp(xvals, data_tstamps_refactored, data_values_refactored)

        yinterp = yinterp.tolist()
        y_interp_array.append(yinterp)

    return y_interp_array


def interpolate_clustered_scenario_data(bus_data, img_tstamps, data_o_inter):
    y_interp_array = []

    for data in data_o_inter:
        data_tstamps = bus_data[data]["timestamps"]
        data_values = bus_data[data]["values"]

        idx_min, idx_max = get_closest_value_clustered_idx(data_tstamps, img_tstamps)

        # data_tstamps_refactored = data_tstamps[idx_min:idx_max]
        # data_values_refactored = data_values[idx_min:idx_max]

        xvals = img_tstamps[idx_min:idx_max]
        yinterp = np.interp(xvals, data_tstamps, data_values)

        yinterp = yinterp.tolist()
        y_interp_array.append(yinterp)

    return y_interp_array, idx_min, idx_max


def main():
    # define path to the created  video by img2vid.py
    vid_path = "change_path/<location>_video.mp4"
    # base data path
    data_path = "data"
    data_place = "Gaimersheim"
    date_of_recording = "20180810"  # date in format YYYYMMDDqq
    time_of_recording = "150607"  # time in format HHMMSS
    camera = "cam_front_center"

    # path to clustered bus data
    cluster_scenario_path = os.path.join("Clustering_approach/Results/driving-scenarios", data_place, "json_files",
                                         "weka.clusterers.SimpleKMeans_scenario_1.json")

    cam_data_path = os.path.join(data_path, data_place, date_of_recording + "_" + time_of_recording, "camera", camera)

    cam_images_files_json = glob.glob(cam_data_path + "/*.json")
    cam_images_files_json.sort()

    img_tstamps = extract_image_timestamps(cam_images_files_json)
    img_tstamps_num = np.arange(0, len(img_tstamps)).tolist()
    bus_data_dict = load_cluster_scenario_data(cluster_scenario_path)

    data_o_interest = ["accelerator_pedal", "brake_pressure", "steering_angle_calculated", "vehicle_speed"]

    # uncomment to save video corresponding to each scenario
    # vid_out = cv.VideoWriter( 'change_path' + camera + "_" +
    # date_of_recording + "_" + time_of_recording + "weka.clusterers.SimpleKMeans_scenario_1.mp4",
    # cv.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

    for episode in bus_data_dict.keys():
        yinterp_array, idx_min, idx_max = interpolate_clustered_scenario_data(bus_data_dict[episode], img_tstamps,
                                                                              data_o_interest)

        # num_frames = cv_frame_count(vid_path)
        # print("Number of Frames: ", num_frames)

        val_tmp = []

        num_subp = len(yinterp_array)
        for i in range(num_subp):
            val_tmp.append([])

        fig, axs = plt.subplots(num_subp, 1)

        fig.canvas.draw()
        plt.show(block=False)

        if isinstance(vid_path, str):
            assert os.path.isfile(vid_path)
            cap = cv.VideoCapture(vid_path)
            cap.set(cv.CAP_PROP_POS_FRAMES, idx_min)
            assert cv_current_frame(cap) == idx_min
            assert cap.isOpened()

        while (cap.isOpened() and cap.get(cv.CAP_PROP_POS_FRAMES) < idx_max):

            # fig.clf()
            ret, frame = cap.read()

            if ret == True:
                # plt.cla()
                frame = cv.resize(frame, (1280, 720))
                cv.imshow("Frame", frame)

                # vid_out.write(frame)

                for i in range(len(yinterp_array)):
                    val_tmp[i].append([yinterp_array[i].pop(0)])
                    # val_tmp[i] = yinterp_array[i].pop(0)
                    # img_tstamps_tmp = img_tstamps_num.pop(0)

                if len(data_o_interest) > 1:
                    for i in range(len(yinterp_array)):
                        txt = data_o_interest[i] + " - current value: {:.2f}"
                        axs[i].set_title(txt.format(val_tmp[i][-1][0]))
                        axs[i].plot(val_tmp[i], "tab:blue")
                        fig.canvas.draw()

                else:
                    txt = data_o_interest[0] + " - current value: {:.2f}"
                    axs.set_title(txt.format(val_tmp[0][-1][0]))
                    axs.plot(val_tmp[0], "tab:blue")
                    fig.canvas.draw()

                k = cv.waitKey(25)
                if k & 0xFF == ord('n'):
                    cap.release()
                    plt.close(fig)
                    break
                elif k & 0xFF == ord('q'):
                    cap.release()
                    return



            else:
                break

        cap.release()

        plt.close(fig)

        cv.destroyAllWindows()

    # vid_out.release()

    # play_vid(vid_path)


if __name__ == "__main__":
    main()
