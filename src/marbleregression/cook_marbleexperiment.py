
import numpy as np

import torch
import argparse
import pickle
from scipy import signal
import pandas as pd



def quaternion_from_rotation_matrix(Q):
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    t = torch.trace(Q)
    r = (1.+ t)**.5
    w = .5 * r

    x = torch.sign(Q[2,1]-Q[1,2]) * torch.abs(.5 * (1.+Q[0,0]-Q[1,1]-Q[2,2])**.5)
    y = torch.sign(Q[0,2]-Q[2,0]) * torch.abs(.5 * (1.-Q[0,0]+Q[1,1]-Q[2,2])**.5)
    z = torch.sign(Q[1,0]-Q[0,1]) * torch.abs(.5 * (1.-Q[0,0]-Q[1,1]+Q[2,2])**.5)

    return torch.tensor([w,x,y,z])

def lin_rot_from_htransform(H):
    lin = H[:3,3]
    rot = quaternion_from_rotation_matrix(H[:3,:3])
    return torch.cat([lin, rot])

def collate_fn_padd(batch):
    '''
    MB: got this from here: https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/13
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    device = "cpu"
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
    ## padd
    batch = [ torch.Tensor(t).to(device) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0).to(device)
    return batch, lengths, mask



def cook_data(data):

    identifiers = [d['identifier'] for d in data]
    ys = np.expand_dims(np.array([float(d["marbles"]) for d in data]),1)


    # modality-wise massaging the data into the feature form we want

    fts = [d["ft"] for d in data]
    visions = [d["vision"] for d in data]
    # sound_spectra = [signal.welch(d["sound_samples"], d["sound_samplerate"], nperseg=64, nfft=256)[1] for d in data]
    # sound_spectograms_alloutputs = [signal.spectrogram(d["sound_samples"], d["sound_samplerate"], nperseg=64,nfft=256) for d in data]
    sound_spectograms_alloutputs = [signal.spectrogram(d["sound_samples"], d["sound_samplerate"], nperseg=256,nfft=256) for d in data]
    sound_spectograms = [d[2].transpose() for d in sound_spectograms_alloutputs]
    sound_times = [d[1] for d in sound_spectograms_alloutputs]

    #sound_times = np.array(sound_times)


    ### align time-series by making them pandas dataframes and interpolating based on timestamps (using sound timestamps
    ### as the master timestamp)
    xs_sound=[]
    xs_ft=[]
    xs_vision =[]
    print("Flattening images from shape",visions[0].shape)

    cutoff = 30

    for i_data in range(len(data)):


        df_sound = pd.DataFrame(data=sound_spectograms[i_data], index = sound_times[i_data])
        df_ft = pd.DataFrame(data=fts[i_data], index = data[i_data]["ft_timestamps"])

        vision_timestamps = data[i_data]["vision_timestamps"]

        #vision_flat = np.reshape(visions[i_data], (-1, 120*160*3))
        vision = visions[i_data]
        vision = vision[::2]
        ts_vision = vision_timestamps[::2]
        # df_vision = pd.DataFrame(data=vision_flat[::2], index = vision_timestamps[::2])

        # ts_vision = df_vision.index

        ts_ft = df_ft.index
        df_ft = df_ft.reindex(list(ts_vision) + ts_ft.to_list()).sort_index()
        df_ft = df_ft[~df_ft.index.duplicated()] # remove duplicates
        df_ft = df_ft.interpolate()
        df_ft = df_ft.reindex(ts_vision).sort_index()

        ts_sound = df_sound.index
        df_sound = df_sound.reindex(list(ts_vision) + ts_sound.to_list()).sort_index()
        df_sound = df_sound[~df_sound.index.duplicated()]  # remove duplicates
        df_sound = df_sound.interpolate()
        df_sound = df_sound.reindex(ts_vision).sort_index()

        df_ft = df_ft.fillna(0.)
        #df_vision = df_vision.fillna(0.)
        df_sound = df_sound.fillna(0.)

        xs_sound.append(df_sound.to_numpy())
        xs_ft.append(df_ft.to_numpy())
        # xs_vision.append(df_vision.to_numpy())
        xs_vision.append(vision)

    # xs_sound = df_sound.to_numpy()
    # xs_ft = df_ft.to_numpy()
    # xs_poses = df_poses.to_numpy()

    # df_ee_position = trial['data']['/ee/position']
    # ts_ee_position = df_ee_position.index
    # df_ee_position = df_ee_position.reindex(ts + ts_ee_position).sort_index()
    # df_ee_position = df_ee_position.interpolate()
    # df_ee_position = df_ee_position.reindex(ts).sort_index()




    # sample_rate, samples = d["sample_rate"], d["samples"]
    # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=64,
    #                                                      nfft=256)  # , noverlap=60)
    # f, pxx = signal.welch(samples, sample_rate, nperseg=64, nfft=256)  # , noverlap=60) # spectrum
    # i_class = classnames.index(d["class"])
    # ys.append(i_class)
    # speeds.append(float(d["speed"]))
    # xs.append(pxx)







    # pad data for lstm on time-series with different lengths
    # xs_ft = collate_fn_padd(fts)
    # xs_sound = collate_fn_padd(sound_spectograms)
    # xs_pose = collate_fn_padd(poses)

    xs_ft = collate_fn_padd(xs_ft)
    xs_sound = collate_fn_padd(xs_sound)
    xs_vision = collate_fn_padd(xs_vision)

    return identifiers, ys, xs_ft, xs_sound, xs_vision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_pickle", help="direcly load from pickle",
                        action="store_false")
    args = parser.parse_args()

    data_dir = "/home/manuelbaum/Data/audio_ip"
    if args.load_pickle:
        print("loading pickle")
        data = pickle.load(open(data_dir+"/audio_ip.pkl", "rb"))
    else:
        print("loading without pickle")
        data = load_data(data_dir)
        pickle.dump(data, open(data_dir+"/audio_ip.pkl", "wb"))

    classes, ys, xs_ft, xs_sound, xs_pose = cook_data(data)

    # print(classes)
    # print(ys)
    # print(xs_ft)
    # print(xs_sound[0].shape)

    print(xs_pose[0].shape)
    print(xs_ft[0].shape)
    print(xs_sound[0].shape)

if __name__ == "__main__":
    main()