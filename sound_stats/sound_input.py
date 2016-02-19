from __future__ import division, print_function
import os
import h5py
import uuid
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from lasp.timefreq import (gaussian_stft, bandpass_timefreq,
                           define_f_bands, log_spectrogram)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SoundData(object):

    _SAVE_ATTRS = ["chunk_size", "increment"]

    def __init__(self, filename, batch_size=128, chunk_size=None, increment=None):
        """
        Creates an object useful as input to a keras model for studying
        sound statistics. Each subclass of SoundData will define a seperate
        transformation that can be done to the sound.
        :param filename: path to output hdf5 file. Loads data from file if it exists
        :param batch_size: size of the batches as stored in the hdf5 file
        :param chunk_size: duration of chunks (default is no chunking)
        :param increment: step size for chunking (default is 0.5*chunk_size, if applicable)

        :returns: an instance of SoundData
        """

        self.filename = filename
        self.batch_size = batch_size
        if os.path.exists(filename):
            self.load()
        else:
            with h5py.File(filename):
                pass
            self.chunk_size = chunk_size
            if (self.chunk_size is not None) and (increment is None):
                increment = int(0.5 * self.chunk_size)
            self.increment = increment
            self.save()

    @property
    def X(self):

        with h5py.File(self.filename, "r") as hf:
            if "data0" in hf:
                return hf["data0"]
            else:
                return

    @property
    def y(self):
        with h5py.File(self.filename, "r") as hf:
            if "data1" in hf:
                return hf["data1"]
            else:
                return

    @staticmethod
    def load_waveform(waveform, samplerate=16000):

        if isinstance(waveform, tuple):
            waveform = list(waveform)
            for ii, val in enumerate(waveform):
                waveform[ii], fs = self.load_waveform(waveform, samplerate=samplerate)
        elif isinstance(waveform, str):
            logger.debug("Loading from %s" % waveform)
            fs, waveform = wavfile.read(waveform)
        else:
            fs = samplerate

        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        return waveform, fs

    def chunk_data(self, data):

        if self.chunk_size is not None:
            chunks = list()
            duration = data.shape[1]
            for start in range(0, duration, self.increment):
                inds = range(start, start + self.chunk_size)

                # If the segment extends beyond the duration of the sound,
                # use just the last full segment that fits
                if inds[-1] >= duration:
                    inds = range(duration - self.chunk_size, duration)

                chunks.append(data[:, inds].ravel())
                if inds[-1] == (duration - 1):
                    break
        else:
            chunks = [data.ravel()]

        return np.vstack(chunks)

    def compute_and_store(self, waveforms, *args, **kwargs):
        """
        Computes the transformation on each sound in waveforms and stores them in a dataset.
        :param waveforms: list of sound waveforms or .wav files
        :param samplerate: default sample rate (Hz) to use if waveform is an array
        :param append: append data to dataset or overwrite (default False because not yet implemented)
        :return list of hdf5 datasets
        """

        samplerate = kwargs.pop("samplerate", 16000)
        # append = kwargs.pop("append", False)
        append = False
        start = 0
        with h5py.File(self.filename, "a") as hf:
            logger.info("Starting loop through %d waveforms" % len(waveforms))
            for ii, waveform in enumerate(waveforms):
                waveform, fs = self.load_waveform(waveform, samplerate)
                kwargs["samplerate"] = fs
                logger.info("%d) Computing transform" % ii)
                outputs = self.compute(waveform, *args, **kwargs)

                if isinstance(outputs, tuple):
                    outputs = list(outputs)
                else:
                    outputs = [outputs]

                if ii == 0:
                    datasets = list()
                    for jj, output in enumerate(outputs):
                        if self.chunk_size is None:
                            ncols = np.prod(output.shape)
                        else:
                            ncols = output.shape[0] * self.chunk_size
                        ds = hf.create_dataset("data%d" % jj,
                                               (1, ncols),
                                               maxshape=(None, ncols),
                                               chunks=(self.batch_size, ncols))
                        datasets.append(ds)

                for ds, output in zip(datasets, outputs):
                    output = self.chunk_data(output)
                    nrows = output.shape[0]
                    ds.resize(start + nrows, 0)
                    ds[start: start+nrows] = output

                start += nrows

        return datasets

    def load(self):
        """
        Loads data from the hdf5 file
        """

        or_none = lambda x: x if x != "none" else None
        with h5py.File(self.filename, "r") as hf:
            for attr, val in hf.attrs.items():
                setattr(self, attr, or_none(val))

    def save(self):
        """
        Saves all attributes in _SAVE_ATTRS and _save_attrs to the hdf5 file
        """

        or_none = lambda x: x if x is not None else "none"
        with h5py.File(self.filename, "a") as hf:
            for attr in self._SAVE_ATTRS + self._save_attrs:
                hf.attrs[attr] = or_none(getattr(self, attr, None))

    def compute(self):

        raise NotImplementedError()


class Spectrogram(SoundData):

    _save_attrs = ["frequencies", "time"]

    def __init__(self, *args, **kwargs):

        super(Spectrogram, self).__init__(*args, **kwargs)
        self.frequencies = None
        self.time = None

    def compute(self, waveform, window_length, increment, samplerate=16000,
                min_freq=0, max_freq=None, nstd=6, offset=50):

        self.time, self.frequencies, spec = gaussian_stft(waveform,
                                                          samplerate,
                                                          window_length,
                                                          increment,
                                                          min_freq=min_freq,
                                                          max_freq=max_freq,
                                                          nstd=nstd)[:3]
        return log_spectrogram(np.abs(spec), offset=offset)

    def plot(self, data=None):

        if data is None:
            data = self.data

        plt.imshow(data,
                   aspect="auto",
                   origin="lower",
                   extent=[self.time[0], self.time[-1],
                           self.frequencies[0], self.frequencies[-1]])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Log spectrogram")


class MeanCenterSpectrogram(Spectrogram):

    _save_attrs = ["frequencies", "time", "mean", "global_sub"]

    def __init__(self, *args, **kwargs):

        self.global_sub = kwargs.get("global_sub", False)
        self.mean = None
        super(MeanCenterSpectrogram, self).__init__(*args, **kwargs)

    def compute(self, *args, **kwargs):

        output = super(MeanCenterSpectrogram, self).compute(*args, **kwargs)
        self.mean = np.mean(output, axis=1)

        return output - self.mean.reshape(-1, 1)


class RatioMask(SoundData):

    _save_attrs = ["frequencies", "time", "alpha"]

    def __init__(self, *args, **kwargs):

        self.alpha = kwargs.get("alpha", 1.0)
        self.frequencies = None
        self.time = None
        super(RatioMask, self).__init__(*args, **kwargs)

    def compute(self, waveforms, window_length, increment, min_freq=0, max_freq=None, nstd=6):

        signal, stimulus = waveforms

        self.time, self.frequencies, spec = gaussian_stft(signal,
                                                          self.sound.samplerate,
                                                          window_length,
                                                          increment,
                                                          min_freq=min_freq,
                                                          max_freq=max_freq,
                                                          nstd=nstd)[:3]
        stim_spec = gaussian_stft(stimulus,
                                  self.sound.samplerate,
                                  window_length,
                                  increment,
                                  min_freq=min_freq,
                                  max_freq=max_freq,
                                  nstd=nstd)[2]

        return stim_spec, np.abs(spec) ** self.alpha / np.abs(stim_spec) ** self.alpha


class Cochleagram(SoundData):

    def __init__(self, *args, **kwargs):

        super(Cochleagram, self).__init__(*args, **kwargs)


class MelSpectrogram(SoundData):

    def __init__(self, *args, **kwargs):

        super(MelSpectrogram, self).__init__(*args, **kwargs)


class MFCC(SoundData):

    def __init__(self, *args, **kwargs):

        super(MFCC, self).__init__(*args, **kwargs)


def temporary_file():

    return os.path.join("/tmp", uuid.uuid4().get_hex())


def to_dataset(sound_inputs, chunk_size=None, increment=None):
    """ Take a list of SoundData instances and convert them to an hdf5 dataset for use with Keras models
    :param sound_inputs: a list of SoundData instances
    :param chunk_size: the duration of a chunk (default None means no chunking)
    :param increment: the number of time points to increment (default None does 0.5 * chunk_size)

    returns hdf5 file containing dataset with the sound_inputs
    """

    if chunk_size is None:
        nrows = len(sound_inputs)
        ncols = np.prod(sound_inputs[0].shape)
    else:
        get_nchunks = lambda dur: np.ceil(float(dur) / int(self.chunk_size * self.stride))
        # Compute the total number of vectorized samples in the list of sound_inputs
        nrows = np.sum([get_nchunks(s.sound.annotations["data_shape"][1]) for s in sound_inputs])
        ncols = sound_inputs[0].shape[0] * chunk_size

    mean = np.zeros(data_shape[1])
    std = np.ones(data_shape[1])
    if self.zscore:
        for s in sound_inputs:
            data = s.data.ravel()
            s.clear_cache()
            mean += data
            std += data ** 2
        std = np.sqrt(std / len(sound_inputs) - mean ** 2 / len(sound_inputs))
        mean = mean / len(sound_inputs)


    with h5py.File(self.ds_filename, "w") as hf:
        ds = hf.create_dataset("input",
                               data_shape,
                               chunks=(batch_size, data_shape[1]))
        data = list()
        start = 0
        for s in sound_inputs:
            data.append(s.data.ravel())
            s.clear_cache()
            if len(data) == batch_size:
                ds[start: start + batch_size] = (np.vstack(data) - mean) / std
                start += batch_size
                data = list()
        if len(data):
            ds[start: start + len(data)] = np.vstack(data)

    return self.ds_filename


def chunk_sound(sound, chunk_size, stride=0.5):
    """
    Vectorizes sound data
    :param sound: a sound_input object
    :param chunk_size: number of time points per chunk
    :param stride: fraction of chunk_size to stride
    :return: a list of vectorized data
    """

    duration = sound.data.shape[1]

    chunks = list()
    # Loop through all segments of the sound
    for chunk_start in range(0, duration, int(chunk_size * stride)):
        inds = range(chunk_start, chunk_start + chunk_size)

        # If the segment extends beyond the duration of the sound,
        # use just the last full segment that fits
        if inds[-1] >= duration:
            inds = range(duration - chunk_size, duration)

        chunks.append(sound.data[:, inds].ravel())
        if inds[-1] == (duration - 1):
            break

    sound.clear_cache()

    return chunks
