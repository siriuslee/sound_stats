import h5py
import os
import numpy as np
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from . import sound_input
from . import embeddings

def filters_to_weights(filename):

    tmp = loadmat(filename, variable_names=["strf"])
    filters = [filt[0] for filt in tmp["strf"]["filter"]]

    nfilters = len(filters)
    nfreq, ndelays = filters[0].shape
    weights = [np.array(filters).reshape((nfilters, 1, nfreq, ndelays)),
               np.zeros((nfilters,))]

    return weights

def stimuli_from_mat(directory, train=True):

    stim_file = os.path.join(directory, "stimuli.mat")
    time_frequency_file = os.path.join(directory, "time_frequency.mat")
    tmp = loadmat(stim_file, squeeze_me=True, variable_names=["signal", "stimulus",
                                                              "trainingSet", "testingSet",
                                                              "wavStimIndex", "Preprocessing"])

    # Get data from the mat file
    training_inds = tmp["trainingSet"]
    testing_inds = tmp["testingSet"]
    if train:
        inds = training_inds
    else:
        inds = testing_inds
    stim_inds = tmp["wavStimIndex"]
    signal = tmp["signal"]
    stimulus = tmp["stimulus"]
    fs = int(tmp["Preprocessing"]["fs"])

    # Get time-frequency parameters as well
    params = dict()
    params["samplerate"] = fs
    with h5py.File(time_frequency_file, "r") as hf:
        getval = lambda ss: hf["TimeFrequency"][ss][()].squeeze()
        params["window_length"] = getval("windowLength") / fs
        params["increment"] = 1.0 / getval("specFs")
        params["min_freq"] = getval("lowFrequency")
        params["max_freq"] = getval("highFrequency")
        params["offset"] = -getval("dBFloor")

    # Get list of signals and stimuli
    signals = list()
    stimuli = list()
    for ii in inds:
        inds = stim_inds == ii
        signals.append(signal[inds])
        stimuli.append(stimulus[inds])

    return signals, stimuli, params


class NoiseFilterNetwork(embeddings.TimeDelayConvolutionNetwork):

    def __init__(self, detection_length=None,
                 reconstruction_length=None,
                 reconstruction_delay=0,
                 stride=0.5,
                 output_type="mask",
                 **kwargs):

        super(NoiseFilterNetwork, self).__init__(input_size=detection_length,
                                                 stride=stride,
                                                 output_size=reconstruction_length,
                                                 output_delay=reconstruction_delay,
                                                 **kwargs)
        self.output_type = output_type

    def predict(self, sound_inputs, **kwargs):

        stride = 1.0 / self.chunk_size
        outputs = super(NoiseFilterNetwork, self).predict(sound_inputs, stride=stride, **kwargs)
        if self.output_type == "mask":
            logistic = lambda x: (1 + np.exp(-x)) ** -1
            outputs = [logistic(out) for out in outputs]

        return outputs


### NOTES:
### If the reconstruction delays range from a0 to aM and the encoding delays range from b0 to bM, then the total number of time points needed for a single batch is (aM + bM) - (a0 + b0) + (batch_size - 1). Traditionally this has been (0 + 99) - (-99 + 0) + (batch_size - 1) = 197 + batch_size.
