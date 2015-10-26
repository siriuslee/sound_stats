from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from lasp.timefreq import gaussian_stft, bandpass_timefreq, define_f_bands, log_spectrogram
from neosound.sound import *

class SoundInput(object):

    _data_name = None
    _save_attrs = []

    def __init__(self, filename=None, tag=""):
        """
        Creates an object useful as input to an embedding model for studying sound statistics. Each subclass of SoundInput will define a seperate transformation that can be done to the sound. Data is stored in an h5 file using neosound's HDF5Store class.
        :param filename: path to a .wav file
        """


        if filename is not None:
            self.sound = Sound(filename).to_mono()
        else:
            self.sound = None

        self._cached_data = None
        self._filename = None
        self.tag = tag

    @classmethod
    def load(cls, id_, filename=None, store=None, full=False, tag=""):
        """
        Loads an object from the hdf5 file
        :param id_: unique sound ID to load. Use the store's query methods to find it if needed.
        :param filename: path to the h5 file. Creates an HDF5Store for that path.
        :param store: an instance of HDF5Store. Overrides the filename argument
        """

        if store is None:
            store = HDF5Store(filename, read_only=True)

        # Extract data from the sound store
        waveform = store.get_data(id_)
        annotations = store.get_annotations(id_)
        cls_attrs = dict()
        for ss in cls._save_attrs:
            name = "%s%s_%s" % (cls._data_name, tag, ss)
            val = store.get_data(id_, name=name)
            if val is not None:
                cls_attrs[ss] = val

        obj = cls()
        obj.sound = Sound(waveform, samplerate=float(annotations["samplerate"])*hertz)
        obj.sound.id = id_
        obj.sound.annotate(**annotations)
        if full:
            obj._cached_data = store.get_data(id_, name="%s%s" % (cls._data_name, tag))
        for ss, val in cls_attrs.iteritems():
            setattr(obj, ss, val)

        obj._filename = store.filename

        return obj

    def _load_data(self):

        store = HDF5Store(self._filename, read_only=True)
        self._cached_data = store.get_data(self.sound.id, name="%s%s" % (self._data_name, self.tag))

    @property
    def data(self):

        if self._cached_data is None:
            self._load_data()

        return self._cached_data

    def save(self, filename=None, store=None, overwrite=False):

        if store is None:
            store = HDF5Store(filename)

        store.store_annotations(self.sound.id, **self.sound.annotations)
        store.store_data(self.sound.id, np.asarray(self.sound))

        if self._data_name is not None:
            store.store_data(self.sound.id,
                             self.data,
                             name="%s%s" % (self._data_name, self.tag),
                             overwrite=overwrite)
            for ss in self._save_attrs:
                val = getattr(self, ss, None)
                if val is not None:
                    store.store_data(self.sound.id,
                                     val,
                                     name="%s%s_%s" % (self._data_name, self.tag, ss))

        self._filename = store.filename

    def compute(self):

        raise NotImplementedError()


class Spectrogram(SoundInput):

    _data_name = "spectrogram"
    _save_attrs = ["frequencies", "time"]

    def __init__(self, *args, **kwargs):

        super(Spectrogram, self).__init__(*args, **kwargs)
        self.frequencies = None
        self.time = None

    def compute(self, window_length, increment, min_freq=0, max_freq=None, nstd=6, offset=50):

        self.time, self.frequencies, spec = gaussian_stft(np.asarray(self.sound).squeeze(),
                                                          self.sound.samplerate,
                                                          window_length,
                                                          increment,
                                                          min_freq=min_freq,
                                                          max_freq=max_freq,
                                                          nstd=nstd)[:3]
        self._cached_data = log_spectrogram(np.abs(spec), offset=offset)

    def plot(self):

        plt.imshow(self.data,
                   aspect="auto",
                   origin="lower",
                   extent=[self.time[0], self.time[-1],
                           self.frequencies[0], self.frequencies[-1]])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Log spectrogram")


class Cochleagram(SoundInput):

    def __init__(self, *args, **kwargs):

        super(Cochleagram, self).__init__(*args, **kwargs)


class MelSpectrogram(SoundInput):

    def __init__(self, *args, **kwargs):

        super(MelSpectrogram, self).__init__(*args, **kwargs)


class MFCC(SoundInput):

    def __init__(self, *args, **kwargs):

        super(MFCC, self).__init__(*args, **kwargs)


