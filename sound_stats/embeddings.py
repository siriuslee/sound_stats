import h5py
import logging
import copy
import os
import numpy as np
import theano
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, AutoEncoder, ActivityRegularization
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers import containers
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from .sound_input import SoundInput

def temporary_file():

    import uuid

    return os.path.join("/tmp", uuid.uuid4().get_hex())

class Embedding(object):

    def __init__(self, model=None, input_shape=None, **kwargs):

        self.model = model
        self.input_shape = input_shape
        self.ds_filename = None

    def create(self, *args, **kwargs):

        raise NotImplementedError()

    def fit(self, sound_inputs, **kwargs):

        pass

    def sounds_to_dataset(self, sound_inputs, batch_size=128):

        raise NotImplementedError()

    def sounds_to_input(self, sound_inputs):

        raise NotImplementedError()

    def output_to_sounds(self, output):

        raise NotImplementedError()

    @classmethod
    def load(cls, filename):

        pass

    def save(self, filename):

        pass


class KerasModel(Embedding):

    _save_attrs = ["input_shape"]

    def __init__(self, zscore=False, **kwargs):

        super(KerasModel, self).__init__(**kwargs)
        self.mean = None
        self.covariance = None
        self.zscore = zscore
        self.zscore_params = list()

    def fit(self, sound_inputs, batch_size=128, **kwargs):

        if isinstance(sound_inputs, list):
            self.zscore_params = list()
            filename = self.sounds_to_dataset(sound_inputs, batch_size=batch_size)
        elif os.path.isfile(sound_inputs):
            filename = sound_inputs
        elif self.ds_filename is not None:
            filename = self.ds_filename
        else:
            IOError("No sound inputs file named %s" % sound_inputs)

        with h5py.File(filename, "r") as hf:
            data = hf["input"]
            self.model.fit(data, data, batch_size=batch_size, **kwargs)

    def get_encoder_layer(self, layer):

        ae = [ll for ll in self.model.layers if isinstance(ll, AutoEncoder)][0]
        layers = [ll for ll in ae.encoder.layers if hasattr(ll, "W")]

        try:
            return layers[layer]
        except IndexError:
            raise IndexError("Model only has %d encoder layers with weights" % len(layers))

    def get_output(self, sound_inputs, layer):

        ll = self.get_encoder_layer(layer)
        get_feature = theano.function([self.model.layers[0].input],
                                      ll.get_output(train=False),
                                      allow_input_downcast=True)
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

        return self.mean, self.covariance

    def get_filters(self, layer):

        for ii in range(layer + 1):
            ll = self.get_encoder_layer(ii)
            if ii == 0:
                filters = ll.get_weights()[0]
            else:
                filters = np.dot(filters, ll.get_weights()[0])

        return [filters[:, ii].reshape(self.input_shape) for ii in range(filters.shape[1])]

    def sample(self, layer, mean=None, covariance=None, nsamples=1):

        if mean is None:
            mean = self.mean
        if covariance is None:
            covariance = self.covariance

        ae = [ll for ll in self.model.layers if isinstance(ll, AutoEncoder)][0]
        input_layer = ae.encoder.layers[layer]
        layers = ae.encoder.layers[layer + 1:] + ae.decoder.layers
        layers = copy.deepcopy(layers)

        new_model = Sequential()
        # Have to get rid of the "previous" layer for the new first layer
        delattr(layers[0], "previous")
        layers[0].set_input_shape((input_layer.ouput_dim,))
        for ll in layers:
            new_model.add(ll)
        new_model.compile(optimizer="sgd", loss="mean_squared_error")
        data = np.random.multivariate_normal(mean, covariance, nsamples)

        return self.output_to_sounds(new_model.predict(data))

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

    def __init__(self, **kwargs):

        super(DeepNetwork, self).__init__(**kwargs)


    def sounds_to_dataset(self, sound_inputs, batch_size=128):

        self.ds_filename = temporary_file()
        data_shape = (len(sound_inputs), np.prod(self.input_shape))

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

    def sounds_to_input(self, sound_inputs):

        return np.vstack([s.data.ravel() for s in sound_inputs])

    def output_to_sounds(self, output):

        return [out.reshape(self.input_shape) for out in output]

    def create(self, input_shape,
               layer_sizes,
               noise_sigma=0.1,
               activation="relu",
               normalize_output=False,
               l1_penalty=0,
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

        if isinstance(activation, list):
            if len(activation) != len(layer_sizes):
                raise ValueError("If activation is a list it must be of the same length as layer sizes")
        else:
            activation = [activation] * len(layer_sizes)


        input_dim = np.prod(self.input_shape)

        if output_activation is None:
            output_activation = activation

        nlayers = len(layer_sizes)

        # Create the encoding layers
        encoding_layers = layer_sizes[:int(nlayers / 2) + 1]
        encoder = list()
        for ii, ls in enumerate(encoding_layers):
            if ii == 0:
                layer = Dense(ls, input_dim=input_dim, activation=activation[ii], **kwargs)
            else:
                layer = Dense(ls, activation=activation[ii], **kwargs)
            encoder.append(layer)
            if l1_penalty > 0:
                encoder.append(ActivityRegularization(l1=l1_penalty))
            if dropout > 0:
                encoder.append(Dropout(dropout))
            if normalize_output:
                encoder.append(BatchNormalization())

        encoder = containers.Sequential(encoder)

        # Create the decoding layers
        decoding_layers = layer_sizes[int(nlayers / 2) + 1:]
        decoder = list()
        for ii, ls in enumerate(decoding_layers):
            if ii == 0:
                layer = Dense(ls,
                              input_dim=encoding_layers[-1],
                              activation=activation[len(encoding_layers) + ii],
                              **kwargs)
            else:
                layer = Dense(ls, activation=activation[len(encoding_layers) + ii], **kwargs)
            decoder.append(layer)
            # if dropout > 0:
            #     decoder.append(Dropout(dropout))

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

    def sounds_to_dataset(self, sound_inputs, batch_size=128):
        """
        Convert sound inputs to hdf5 datasets. Each row in the dataset is a vectorized segment of sound.
        :param sound_inputs: list of sound_input objects
        :param batch_size: specifies the number of rows per hdf5 chunk.
        :return: name of hdf5 file containing the dataset
        """

        self.ds_filename = temporary_file() # Get a /tmp filename

        get_nchunks = lambda dur: np.ceil(float(dur) / int(self.chunk_size * self.stride))

        # Compute the total number of vectorized samples in the list of sound_inputs
        nsamples = np.sum([get_nchunks(s.sound.annotations["data_shape"][1]) for s in sound_inputs])
        data_shape = (nsamples, np.prod(self.input_shape))

        mean = np.zeros(data_shape[1])
        std = np.ones(data_shape[1])
        if self.zscore:
            for s in sound_inputs:
                chunks = np.vstack(self.vectorize_sound(s))
                mean += np.sum(chunks, axis=0)
                std += np.sum(chunks ** 2, axis=0)
            std = np.sqrt(std / nsamples - mean ** 2 / nsamples)
            mean = mean / nsamples

        with h5py.File(self.ds_filename, "w") as hf:
            ds = hf.create_dataset("input",
                                   data_shape,
                                   chunks=(batch_size, data_shape[1]))
            data = list()
            start = 0
            for s in sound_inputs:
                chunks = self.vectorize_sound(s)
                # Do I need to store in the same size chunks as specified when the ds was created?
                # Seems like smaller is definitely okay
                ds[start: start + len(chunks)] = (np.vstack(chunks) - mean) / std
                start += len(chunks)

        return self.ds_filename

    def vectorize_sound(self, sound):
        """
        Vectorizes sound data
        :param sound: a sound_input object
        :return: a list of vectorized data
        """

        duration = sound.data.shape[1]

        chunks = list()
        # Loop through all segments of the sound
        for chunk_start in range(0, duration, int(self.chunk_size * self.stride)):
            inds = range(chunk_start, chunk_start + self.chunk_size)

            # If the segment extends beyond the duration of the sound,
            # use just the last full segment that fits
            if inds[-1] >= duration:
                inds = range(duration - self.chunk_size, duration)

            chunks.append(sound.data[:, inds].ravel())
            if inds[-1] == (duration - 1):
                break
        sound.clear_cache()

        return chunks


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

    def fit(self, sound_inputs, sound_outputs=None, batch_size=128, **kwargs):

        if isinstance(sound_inputs, list):
            filename = self.sounds_to_dataset(sound_inputs, sound_outputs=sound_outputs,
                                              batch_size=batch_size)
        elif os.path.isfile(sound_inputs):
            filename = sound_inputs
        elif self.ds_filename is not None:
            filename = self.ds_filename
        else:
            IOError("No sound inputs file named %s" % sound_inputs)

        with h5py.File(filename, "r") as hf:
            inputs = hf["input"]
            outputs = hf["output"]
            self.model.fit(inputs, outputs, batch_size=batch_size, **kwargs)

    def get_output(self, sound_inputs, layer):

        ae = [ll for ll in self.model.layers if isinstance(ll, AutoEncoder)][0]
        get_feature = theano.function([self.model.layers[0].input],
                                      ae.encoder.layers[layer].get_output(train=False),
                                      allow_input_downcast=False)
        data = self.sounds_to_input(sound_inputs)[0]

        return [out for out in get_feature(data)]

    def predict(self, sound_inputs, stride=None, **kwargs):

        data = self.sounds_to_input(sound_inputs, stride=stride)[0]

        print("Testing model")
        output = self.model.predict(data, **kwargs)

        return self.output_to_sounds(output)

    def sounds_to_dataset(self, sound_inputs, sound_outputs=None, batch_size=128):

        if sound_outputs is None:
            sound_outputs = sound_inputs

        self.ds_filename = temporary_file()
        get_nchunks = lambda dur: np.ceil(float(dur) / int(self.chunk_size * self.stride))
        nsamples = np.sum([get_nchunks(s.sound.annotations["data_shape"][1]) for s in sound_inputs])
        input_shape = (nsamples, np.prod(self.input_shape))
        output_shape = (nsamples, self.input_shape[0] * self.output_size)
        with h5py.File(self.ds_filename, "w") as hf:
            ds_input = hf.create_dataset("input",
                                         input_shape,
                                         chunks=(batch_size, input_shape[1]))
            ds_output = hf.create_dataset("output",
                                          output_shape,
                                          chunks=(batch_size, output.shape[1]))
            inputs = list()
            outputs = list()
            start = 0
            for si, so in zip(sound_inputs, sound_outputs):
                duration = si.data.shape[1]
                last_ind = duration - self.output_delay + (self.chunk_size - self.output_size)
                for chunk in xrange(0, duration, int(self.chunk_size * self.stride)):
                    inds = range(chunk, chunk + self.chunk_size)
                    if inds[-1] >= last_ind:
                        inds = range(last_ind - self.chunk_size, last_ind)
                    inputs.append(si.data[:, inds].ravel())
                    output_inds = range(inds[0] + self.output_delay,
                                        inds[0] + self.output_delay + self.output_size)
                    outputs.append(so.data[:, output_inds].ravel())
                    if len(inputs) == batch_size:
                        ds_input[start: start + batch_size] = np.vstack(inputs)
                        ds_output[start: start + batch_size] = np.vstack(outputs)
                        start += batch_size
                        inputs = list()
                        outputs = list()
                    if inds[-1] == (last_ind - 1):
                        break
                si.clear_cache()
                so.clear_cache()

            if len(inputs):
                ds_input[start: start + len(inputs)] = np.vstack(inputs)
                ds_outputs[start: start + len(outputs)] = np.vstack(outputs)

        return self.ds_filename

    def sounds_to_input(self, sound_inputs, sound_outputs=None, stride=None):

        if sound_outputs is None:
            sound_outputs = sound_inputs

        if stride is None:
            stride = self.stride

        inputs = list()
        outputs = list()
        self._input_numbers = list()
        self._input_durations = dict()
        for ii, (si, so) in enumerate(zip(sound_inputs, sound_outputs)):
            duration = si.data.shape[1]
            last_ind = duration - self.output_delay + (self.chunk_size - self.output_size)
            self._input_durations[ii] = duration
            for chunk in xrange(0, duration, int(stride * self.chunk_size)):
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


