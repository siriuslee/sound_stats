import h5py
import os
import numpy as np
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from . import sound_input

def filters_to_weights(filename):

    tmp = loadmat(filename, variable_names=["strf"])
    filters = [filt[0] for filt in tmp["strf"]["filter"]]

    nfilters = len(filters)
    nfreq, ndelays = filters[0].shape
    weights = [np.array(filters).reshape((nfilters, 1, nfreq, ndelays)),
               np.zeros((nfilters,))]

    return weights

def stimuli_to_sound_inputs(directory, output_file=None, output="mask"):

    stim_file = os.path.join(directory, "stimuli.mat")
    time_frequency_file = os.path.join(directory, "time_frequency.mat")
    tmp = loadmat(stim_file, squeeze_me=True, variable_names=["signal", "stimulus",
                                                              "trainingSet", "testingSet",
                                                              "wavStimIndex", "Preprocessing"])

    training_inds = tmp["trainingSet"]
    testing_inds = tmp["testingSet"]
    stim_inds = tmp["wavStimIndex"]
    signal = tmp["signal"]
    stimulus = tmp["stimulus"]
    fs = int(tmp["Preprocessing"]) * sound_input.hertz

    with h5py.File(time_frequency_file, "r") as hf:
        params = dict()
        keys = ["windowLength", "specFs", "lowFrequency", "highFrequency",
                "bandwidth", "dBFloor"]
        for key in keys:
            params[key] = hf["TimeFrequency"][key][()].squeeze()

    for ii in training_inds:
        inds = stim_inds == ii
        spec = sound_input.MeanCenterSpectrogram()
        spec.sound = sound_input.Sound(stimulus, samplerate=fs)
        spec.compute(fft_pts * float(spec.sound.sampleperiod),
                     .001, min_freq=25, max_freq=8000)


### NOTES:
### If the reconstruction delays range from a0 to aM and the encoding delays range from b0 to bM, then the total number of time points needed for a single batch is (aM + bM) - (a0 + b0) + (batch_size - 1). Traditionally this has been (0 + 99) - (-99 + 0) + (batch_size - 1) = 197 + batch_size.

