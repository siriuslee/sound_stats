import h5py
import logging
import os
import numpy as np
import theano
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, AutoEncoder
from keras.layers.noise import GaussianNoise
from keras.layers import containers
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from .sound_input import SoundInput


class Embedding(object):

    def __init__(self, npcs=None, model=None, input_shape=None, **kwargs):

        self.model = model
        self.input_shape = input_shape

    def create(self, *args, **kwargs):

        raise NotImplementedError()

    def fit(self, sound_inputs, **kwargs):

        pass

    @classmethod
    def load(cls, filename):

        pass

    def save(self, filename):

        pass


class KerasModel(Embedding):

    _save_attrs = ["input_shape"]

    def __init__(self, *args, **kwargs):

        super(KerasModel, self).__init__(*args, **kwargs)
        self.mean = None
        self.covariance = None

    def fit(self, sound_inputs, **kwargs):

        data = self.sounds_to_input(sound_inputs)
        self.model.fit(data, data, **kwargs)

    def get_output(self, sound_inputs, layer):

        ae = [ll for ll in self.model.layers if isinstance(ll, AutoEncoder)][0]
        get_feature = theano.function([self.model.layers[0].input],
                                      ae.encoder.layers[layer].get_output(train=False),
                                      allow_input_downcast=False)
        data = self.sounds_to_input(sound_inputs)

        return [out for out in get_feature(data)]

    def predict(self, sound_inputs, **kwargs):

        data = self.sounds_to_input(sound_inputs)
        output = self.model.predict(data, **kwargs)

        return self.output_to_sounds(output)

    def evaluate(self, sound_inputs, **kwargs):

        data = self.sounds_to_input(sound_inputs)

        return self.model.evaluate(data, data, **kwargs)

    def compute_statistics(self, sound_inputs=None, layer=None, outputs=None):

        if outputs is None:
            outputs = self.get_output(sound_inputs, layer)
        outputs = np.vstack(outputs)
        self.covariance = np.cov(outputs, rowvar=False)
        self.mean = np.mean(outputs, axis=0)

    def sample(self, layer, mean=None, covariance=None):

        if mean is None:
            mean = self.mean
        if covariance is None:
            covariance = self.covariance

        ae = [ll for ll in self.model.layers if isinstance(ll, AutoEncoder)][0]
        new_model = Sequential()
        new_model.add([ll for ll in ae.layers[layer:]]) # Not right yet

    @classmethod
    def load(cls, directory):

        weights_file = os.path.join(directory, "weights.h5")
        model_file = os.path.join(directory, "model.json")

        with open(model_file, "r") as f:
            json_string = f.read()

        model = model_from_json(json_string)
        model.load_weights(weights_file)
        params = dict()
        with h5py.File(weights_file, "r") as hf:
            g = hf["model_attr"]
            for key, ds in g.items():
                params[key] = ds[()]

        emb = cls(model=model, **params)

        return emb

    def save(self, directory, overwrite=False):

        if not os.path.exists(directory):
            os.makedirs(directory)

        weights_file = os.path.join(directory, "weights.h5")
        model_file = os.path.join(directory, "model.json")

        if overwrite or not os.path.isfile(weights_file):
            self.model.save_weights(weights_file, overwrite=overwrite)
            with open(model_file, "w") as f:
                f.write(self.model.to_json())
            if self._save_attrs is not None and len(self._save_attrs) > 0:
                with h5py.File(weights_file, "a") as hf:
                    g = hf.create_group("model_attr")
                    for key in self._save_attrs:
                        val = getattr(self, key, None)
                        g.create_dataset(key, data=val)

class DeepNetwork(KerasModel):

    _save_attrs = ["input_shape"]

    def __init__(self, *args, **kwargs):

        super(DeepNetwork, self).__init__(*args, **kwargs)

    def sounds_to_input(self, sound_inputs):

        return np.vstack([s.data.ravel() for s in sound_inputs])

    def output_to_sounds(self, output):

        return [out.reshape(self.input_shape) for out in output]

    def create(self, input_shape,
               layer_sizes,
               noise_sigma=0.1,
               activation="relu",
               dropout=0.5,
               output_activation=None,
               optimizer="adam",
               loss="mean_squared_error",
               **kwargs):

        if isinstance(input_shape, SoundInput):
            self.input_shape = input_shape.data.shape
        elif isinstance(input_shape, np.ndarray):
            self.input_shape = input_shape.shape
        elif isinstance(input_shape, (list, tuple)):
            self.input_shape = input_shape
        else:
            raise ValueError("input_shape is of unknown type: %s" % str(type(input_shape)))

        input_dim = np.prod(self.input_shape)

        if output_activation is None:
            output_activation = activation

        nlayers = len(layer_sizes)

        # Create the encoding layers
        encoding_layers = layer_sizes[:int(nlayers / 2) + 1]
        encoder = list()
        for ii, ls in enumerate(encoding_layers):
            if ii == 0:
                layer = Dense(ls, input_dim=input_dim, activation=activation, **kwargs)
            else:
                layer = Dense(ls, activation=activation, **kwargs)
            encoder.append(layer)
            if dropout > 0:
                encoder.append(Dropout(dropout))
        encoder = containers.Sequential(encoder)

        # Create the decoding layers
        decoding_layers = layer_sizes[int(nlayers / 2) + 1:]
        decoder = list()
        for ii, ls in enumerate(decoding_layers):
            if ii == 0:
                layer = Dense(ls,
                              input_dim=encoding_layers[-1],
                              activation=activation,
                              **kwargs)
            else:
                layer = Dense(ls, activation=activation, **kwargs)
            decoder.append(layer)
            if dropout > 0:
                decoder.append(Dropout(dropout))

        # Add the output layer
        if len(decoder):
            decoder.append(Dense(input_dim, activation=output_activation, **kwargs))
        else:
            decoder.append(Dense(input_dim,
                                 input_dim=encoding_layers[-1],
                                 activation=output_activation,
                                 **kwargs))
        decoder = containers.Sequential(decoder)

        self.model = Sequential()
        if noise_sigma > 0:
            self.model.add(GaussianNoise(noise_sigma, input_shape=(input_dim,)))
        self.model.add(AutoEncoder(encoder, decoder, output_reconstruction=True))
        self.model.compile(loss=loss, optimizer=optimizer)


class TimeConvolutionNetwork(DeepNetwork):

    _save_attrs = ["chunk_size", "stride", "input_shape"]

    def __init__(self, chunk_size=None, stride=0.5, **kwargs):

        super(TimeConvolutionNetwork, self).__init__(**kwargs)
        self.chunk_size = chunk_size
        self.stride = stride
        self._input_numbers = None
        self._input_durations = None

    def sounds_to_input(self, sound_inputs):

        data = list()
        self._input_numbers = list()
        self._input_durations = dict()
        for ii, s in enumerate(sound_inputs):
            duration = s.data.shape[1]
            self._input_durations[ii] = duration
            for chunk in xrange(0, duration, int(self.chunk_size * self.stride)):
                inds = range(chunk, chunk + self.chunk_size)
                if inds[-1] >= duration:
                    inds = range(duration - self.chunk_size, duration)
                data.append(s.data[:, inds].ravel())
                self._input_numbers.append((ii, inds[0]))
                if inds[-1] == (duration - 1):
                    break

        return np.vstack(data)

    def output_to_sounds(self, output):

        sounds = list()
        last_input = -1
        for ii, out in enumerate(output):
            input, chunk = self._input_numbers[ii]
            if input != last_input:
                if ii != 0:
                    sounds.append(s / m)
                s = np.zeros((self.input_shape[0], self._input_durations[input]))
                m = np.zeros_like(s)
                last_input = input
            inds = range(chunk, chunk + self.chunk_size)
            s[:, inds] += out.reshape(self.input_shape)
            m[:, inds] += 1.0
        sounds.append(s / m)

        return sounds

    def create(self, input_shape, *args, **kwargs):

        if isinstance(input_shape, SoundInput):
            self.input_shape = (input_shape.data.shape[0], self.chunk_size)
        elif isinstance(input_shape, np.ndarray):
            self.input_shape = (input_shape.shape[0], self.chunk_size)
        elif isinstance(input_shape, (list, tuple)):
            self.input_shape = (input_shape[0], self.chunk_size)
        else:
            raise ValueError("input_shape is of unknown type: %s" % str(type(input_shape)))

        super(TimeConvolutionNetwork, self).create(self.input_shape, *args, **kwargs)


class TimeDelayConvolutionNetwork(TimeConvolutionNetwork):

    _save_attrs = ["input_size", "stride", "output_size", "output_delay", "input_shape"]

    def __init__(self, input_size=None, stride=0.5, output_size=None, output_delay=0, **kwargs):

        super(TimeDelayConvolutionNetwork, self).__init__(chunk_size=input_size,
                                                          stride=stride,
                                                          **kwargs)
        self.output_delay = output_delay
        if output_size is None:
            self.output_size = self.chunk_size
        else:
            self.output_size = output_size

    def fit(self, sound_inputs, sound_outputs=None, **kwargs):

        input, output = self.sounds_to_input(sound_inputs, sound_outputs)
        self.model.fit(input, output, **kwargs)

    def get_output(self, sound_inputs, layer):

        ae = [ll for ll in self.model.layers if isinstance(ll, AutoEncoder)][0]
        get_feature = theano.function([self.model.layers[0].input],
                                      ae.encoder.layers[layer].get_output(train=False),
                                      allow_input_downcast=False)
        data = self.sounds_to_input(sound_inputs)[0]

        return [out for out in get_feature(data)]

    def predict(self, sound_inputs, **kwargs):

        data = self.sounds_to_input(sound_inputs)[0]

        print("Testing model")
        output = self.model.predict(data, **kwargs)

        return self.output_to_sounds(output)

    def sounds_to_input(self, sound_inputs, sound_outputs=None):

        if sound_outputs is None:
            sound_outputs = sound_inputs

        inputs = list()
        outputs = list()
        self._input_numbers = list()
        self._input_durations = dict()
        for ii, (si, so) in enumerate(zip(sound_inputs, sound_outputs)):
            duration = si.data.shape[1]
            last_ind = duration - self.output_delay + (self.chunk_size - self.output_size)
            self._input_durations[ii] = duration
            for chunk in xrange(0, duration, int(self.stride * self.chunk_size)):
                inds = range(chunk, chunk + self.chunk_size)
                if inds[-1] >= last_ind:
                    inds = range(last_ind - self.chunk_size, last_ind)
                inputs.append(si.data[:, inds].ravel())
                output_inds = range(inds[0] + self.output_delay,
                                    inds[0] + self.output_delay + self.output_size)
                outputs.append(so.data[:, output_inds].ravel())
                self._input_numbers.append((ii, inds[0]))
                if inds[-1] == (last_ind - 1):
                    break

        return np.vstack(inputs), np.vstack(outputs)

    def output_to_sounds(self, output):

        sounds = list()
        last_input = -1
        for ii, out in enumerate(output):
            input, chunk = self._input_numbers[ii]
            if input != last_input:
                if ii != 0:
                    s[m != 0] /= m[m != 0]
                    sounds.append(s)
                s = np.zeros((self.input_shape[0], self._input_durations[input]))
                m = np.zeros_like(s)
                last_input = input
            inds = range(chunk + self.output_delay,
                         chunk + self.output_delay + self.output_size)
            s[:, inds] += out.reshape((self.input_shape[0], self.output_size))
            m[:, inds] += 1.0
        s[m != 0] /= m[m != 0]
        sounds.append(s)

        return sounds


