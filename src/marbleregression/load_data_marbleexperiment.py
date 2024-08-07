import rosbag
import tf
import rospy
import os
import numpy as np
from scipy.io import wavfile
from cv_bridge import CvBridge



def handle_wav_file(filepath):
    """
    Load a wav file into a numpy array
    :param filepath: the filename of the wav file
    :return: a tuple (sample_rate, samples)
    """
    sample_rate, samples = wavfile.read(filepath)
    return sample_rate, samples
def handle_rosbag_file(bag_filename):
    """
    Load a rosbag file and convert force-torque and vision data in the file to be in a format that that easily be cooked for ML
    :param bag_filename: the filename of the rosbag
    :return: a tuple of numpy arrays: (vision, vision_timestamps, ft, ft_timestamps)
    """
    vision = []
    vision_timestamps = []
    with rosbag.Bag(bag_filename, 'r') as bag:

        ### Pulling out ee-position in rlab_origin frame
        cnt = 0
        transformer = tf.TransformerROS()
        for topic, msg, t in bag.read_messages(['/cv_camera_node/image_raw']):

            if cnt == 0:
                initial_timestamp = t

            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            image = np.asarray(cv_image)

            vision.append(image)
            vision_timestamps.append((t - initial_timestamp).to_sec())

            cnt = cnt + 1

        vision = np.stack(vision)
        vision_timestamps = np.array(vision_timestamps)#.reshape(-1, 1)

        ft = []
        ft_timestamps = []
        ### pulling out ft readings
        cnt = 0
        for topic, msg, t in bag.read_messages(['/ham/ft_notool']):
            if cnt == 0:
                initial_timestamp = t

            ft.append(msg.data)
            ft_timestamps.append((t - initial_timestamp).to_sec())

            cnt = cnt + 1

        ft = np.array(ft)
        ft_timestamps = np.array(ft_timestamps)

        return vision, vision_timestamps, ft, ft_timestamps
def load_data(path, is_reduced = False):
    """
    loads experimental data from a directory. assumes the directory contains .bag files with force-torque and vision, and that the directory contains .wav files of the same name that contain sound
    :param path: the directory from which to load the data
    :param is_reduced: potentially create truncated, shorter timeseries for faster compute in debug situations
    :return: a dictionary that contains the dataset
    """
    data = []
    # go over all .bag files in path. if we find one, we will parse it and also try to read a wav file of the same name
    for i_fp, filepath in enumerate(os.listdir(path)):
        # if is_reduced and i_fp > 10:
        #     break

        if not filepath.endswith(".bag"):
            if not filepath.endswith(".wav"):
                print("ignoring a file that is neither wav nor rosbag:", filepath) # we will ignore wav files silently
            continue

        try:
            filename = os.path.basename(filepath)
            id, marbles, year, time = filename[:-4].split("_")
            identifier = filename[:-4]
        except:
            print("Unusually formatted rosbag filename (reason to exit!):", filepath, filename)
            exit()

        # handle rosbag
        vision, vision_timestamps, ft, ft_timestamps = handle_rosbag_file(os.path.join(path,filename))

        # next to this rosbag, there must be a wav file of the same name
        filename_wav = filename[:-4]+".wav"
        sound_samplerate, sound_samples = handle_wav_file(os.path.join(path,filename_wav))

        if is_reduced:
            vision, vision_timestamps, ft, ft_timestamps = vision[:50], vision_timestamps[:50], ft[:50], ft_timestamps[:50]
            sound_samples=sound_samples[:50]


        d = {"marbles": marbles,
             "identifier": identifier,
             "sound_samplerate": sound_samplerate,
             "sound_samples": sound_samples,
             "vision": vision,
             "vision_timestamps": vision_timestamps,
             "ft": ft,
             "ft_timestamps": ft_timestamps}
        data.append(d)

    # assume we have a .wav file with the same name that contains the sound data (we had to pre-convert that using
    # ffmpeg and can not directly pull sound data from the rosbag

    return data