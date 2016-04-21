from __future__ import division
import h5py
import os
import logging
import numpy as np
import time
from scipy.io import loadmat
from scipy.signal import resample
from . import sound_input
from . import embeddings
from lasp.timefreq import gaussian_bandpass_analytic
from keras.layers import core
from keras import models
import keras.backend as K

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def filters_to_weights(filename):

    tmp = loadmat(filename, variable_names=["strf"])
    filters = [filt[0] for filt in tmp["strf"]["filter"]]

    return np.array(filters)

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


def log_spectrogram(spec, offset=80):

    spec /= spec.max()
    spec = np.maximum(spec, 1e-16)
    spec = 20 * np.log10(spec) + offset

    return np.maximum(spec, 0)


class NoiseFilterSounds(sound_input.SoundData):

    _save_attrs = ["frequencies",
                   "bandwidths",
                   "alpha",
                   "increment",
                   "log",
                   "offset",
                   "samplerate",
                   "mean_center"]

    def __init__(self, *args, **kwargs):

        super(NoiseFilterSounds, self).__init__(*args, **kwargs)
        self.frequencies = None
        self.bandwidths = None
        self.alpha = None
        self.increment = None
        self.log = None
        self.offset = None
        self.samplerate = None
        self.mean_center = None

    @property
    def stimulus_spectrogram(self):
        """
        The mean centered noisy stimulus spectrogram
        """

        return self.get_data("data0")

    @property
    def irm(self):
        """
        The ideal ratio mask between noisy spectrogram and clean spectrogram
        """

        return self.get_data("data1")

    @property
    def filterbank(self):
        """
        The filterbank of the noisy stimulus
        """

        return self.get_data("data5")

    @property
    def signal_spectrogram(self):
        """
        The clean signal spectrogram
        """

        return self.get_data("data4")

    def compute(self, waveforms, samplerate, frequencies, bandwidths,
                increment=.01, log=True, offset=50, alpha=1.0,
                mean_center=True, threshold=1e-3):

        signal, stimulus = waveforms

        nsamples = int(len(signal) * (1.0 / samplerate) / float(increment))
        logger.debug("Calling gaussian_bandpass_analytic")
        start = time.time()
        analytic = gaussian_bandpass_analytic(signal, samplerate,
                                              frequencies, bandwidths)
        logger.debug("Call took %3.2f seconds" % (time.time() - start))

        # Compute the spectrogram from the envelope
        logger.debug("Computing signal spectrogram")
        start = time.time()
        envelope = np.abs(analytic)
        signal_spec = resample(envelope, nsamples, axis=1).T
        signal_spec = np.maximum(signal_spec, 0)
        logger.debug("Took %3.2f seconds" % (time.time() - start))

        logger.debug("Calling gaussian_bandpass_analytic")
        start = time.time()
        analytic = gaussian_bandpass_analytic(stimulus, samplerate,
                                              frequencies, bandwidths)
        logger.debug("Call took %3.2f seconds" % (time.time() - start))
        # Compute filterbank and normalize
        filterbank = np.real(analytic)
        filterbank = filterbank * stimulus.std() / filterbank.sum(axis=0).std()
        snr = 20 * np.log10(stimulus.std() / (stimulus - filterbank.sum(0)).std())
        logger.debug("SNR of filterbank is %3.2f" % snr)

        # Compute the spectrogram from the envelope
        logger.debug("Computing stimulus spectrogram")
        start = time.time()
        envelope = np.abs(analytic)
        stimulus_spec = resample(envelope, nsamples, axis=1).T
        stimulus_spec = np.maximum(stimulus_spec, 0)
        logger.debug("Took %3.2f seconds" % (time.time() - start))

        mask = np.zeros_like(signal_spec)
        max_val = stimulus_spec.max()
        inds = stimulus_spec / max_val >= threshold
        mask[inds] = signal_spec[inds] ** alpha / stimulus_spec[inds] ** alpha
        mask = np.maximum(0, np.minimum(1, mask))
        if log:
            signal_spec = log_spectrogram(signal_spec, offset=offset)
            stimulus_spec = log_spectrogram(stimulus_spec, offset=offset)

        if mean_center:
            stimulus_spec -= stimulus_spec.mean(axis=0)

        self.frequencies = frequencies
        self.bandwidths = bandwidths
        self.increment = increment
        self.log = log
        self.offset = offset
        self.samplerate = samplerate
        self.mean_center = mean_center
        self.save()

        return stimulus_spec, mask, signal, stimulus, signal_spec, filterbank.T


class NoiseFilterNetwork(embeddings.Convolution1DAutoEncoder):

    def __init__(self, input_dim,
                 layer_sizes,
                 input_filter_length,
                 noise_sigma=0.1,
                 activation="relu",
                 batch_normalization=True,
                 dropout=0.0,
                 output_activation="sigmoid",
                 output_dim=None,
                 output_filter_length=None,
                 **kwargs):

        super(NoiseFilterNetwork, self).__init__(input_dim,
                                                 layer_sizes,
                                                 input_filter_length,
                                                 noise_sigma=noise_sigma,
                                                 activation=activation,
                                                 batch_normalization=batch_normalization,
                                                 dropout=dropout,
                                                 output_activation=output_activation,
                                                 output_dim=output_dim,
                                                 output_filter_length=output_filter_length,
                                                 **kwargs)

    def filter_noise(self, stimulus_spec, filterbank):

        n = int(filterbank.shape[1] / stimulus_spec.shape[1])
        input_shape = stimulus_spec.shape[1:]
        output_shape = list(input_shape)
        output_shape[0] *= n
        output_shape = tuple(output_shape)
        filterbank = filterbank[:, :output_shape[0], :]

        if not hasattr(self, "_output") or \
           self._output.input_shape != stimulus_spec.shape:

            repeat_input = lambda input, n: K.repeat_elements(input, n, 1)
            repeat_gains = core.Lambda(repeat_input,
                                       input_shape=stimulus_spec.shape[1:],
                                       output_shape=output_shape,
                                       arguments={"n":n})
            sum_features = lambda input: K.sum(input, axis=-1)
            sum_layer = core.Lambda(sum_features,
                                    output_shape=(output_shape[1],))

            self._output = models.Graph()
            self._output.add_input(name="gains", input_shape=input_shape)
            self._output.add_input(name="filterbank",
                                   input_shape=filterbank.shape[1:])
            self._output.add_node(repeat_gains, name="upsample_gains",
                                  input="gains")
            self._output.add_node(sum_layer,
                                  name="sum",
                                  inputs=["upsample_gains", "filterbank"],
                                  merge_mode="mul",
                                  create_output=True)
            self._output.compile(optimizer="sgd", loss={"sum": "mean_squared_error"})

        gains = self.predict(stimulus_spec)

        return self._output.predict({"gains": gains, "filterbank": filterbank})["sum"]


    def set_weights(self, filename, subsample=1, freeze=True):

        weights = filters_to_weights(filename)
        # Subsample but leave as is
        forward_weights = weights[:, :, ::subsample]
        # Change to nbands x nunits x ndelays
        reverse_weights = forward_weights.transpose((1, 0, 2))
        nunits, nbands, ndelays = forward_weights.shape
        # Add a single last dimension
        forward_weights = forward_weights.reshape(forward_weights.shape + (1,))
        reverse_weights = reverse_weights.reshape(reverse_weights.shape + (1,))
        for layer in self.layers:
            if isinstance(layer, embeddings.Convolution1D):
                layer.set_weights((forward_weights, np.zeros(nunits)))
                layer.trainable = not freeze
                break
        for layer in reversed(self.layers):
            if isinstance(layer, embeddings.Convolution1D):
                layer.set_weights((reverse_weights, np.zeros(nbands)))
                layer.trainable = not freeze
                break
