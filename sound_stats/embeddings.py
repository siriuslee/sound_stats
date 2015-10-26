import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, AutoEncoder
from keras.layers.noise import GaussianNoise
from keras.layers import containers
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.decomposition import RandomizedPCA

from .sound_input import SoundInput


class Embedding(object):

    def __init__(self, npcs=None):

        if npcs is not None:
            self._pca = RandomizedPCA(n_components=npcs)
        else:
            self._pca = None

    def create(self):

        raise NotImplementedError()

    def fit(self):

        pass

    def evaluate(self):

        pass

    @classmethod
    def load(cls):

        pass

    def save(self):

        pass


class KerasModel(Embedding):

    def __init__(self, *args, **kwargs):

        super(KerasModel, self).__init__(*args, **kwargs)


class DeepNetwork(KerasModel):

    def __init__(self, *args, **kwargs):

        super(DeepNetwork, self).__init__(*args, **kwargs)


    def fit(self, sound_inputs, **kwargs):

        print("Extracting data")
        data = np.vstack([s.data.ravel() for s in sound_inputs])
        if self._pca is not None:
            print("Computing PCA")
            data = self._pca.fit_transform(data)
        print("Fitting model")
        self.model.fit(data, data, **kwargs)

    def predict(self, sound_inputs, **kwargs):

        print("Extracting data")
        data = np.vstack([s.data.ravel() for s in sound_inputs])
        if self._pca is not None:
            print "Multiplying by PCs"
            data = self._pca.transform(data)

        print("Testing model")
        output = self.model.predict(data, **kwargs)

        if self._pca is not None:
            print "Inverting PCA"
            output = self._pca.invert_transform(output)

        return [out.reshape(s.data.shape) for s, out in zip(sound_inputs, output)]

    def create(self, input, layer_sizes, noise_sigma=0.1, activation="relu", output_activation=None,
               optimizer="sgd", loss="mean_squared_error", **kwargs):

        if isinstance(input, SoundInput):
            input_dim = np.prod(input.data.shape)
        elif isinstance(input, np.ndarray):
            input_dim = np.prod(input.shape)
        elif isinstance(input, (list, tuple)):
            input_dim = np.prod(input)
        else:
            raise ValueError("input is of unknown type: %s" % str(type(input)))

        if output_activation is None:
            output_activation = activation

        nlayers = len(layer_sizes)

        # Create the encoding layers
        encoding_layers = layer_sizes[:int(nlayers / 2) + 1]
        encoder = [Dense(encoding_layers.pop(0),
                         input_dim=input_dim,
                         activation=activation,
                         **kwargs)]
        encoder += [Dense(ll, activation=activation, **kwargs) for ll in encoding_layers]
        encoder = containers.Sequential(encoder)

        # Create the decoding layers
        decoding_layers = layer_sizes[int(nlayers / 2) + 1:]
        decoder = list()
        if len(decoding_layers) > 0:
            decoder = [Dense(decoding_layers.pop(0),
                             input_dim=encoding_layers[-1],
                             activation=activation,
                             **kwargs)]
            decoder += [Dense(ls, activation=activation, **kwargs) for ls in decoding_layers]
            # Add the output layer
            decoder.append(Dense(input_dim, activation=output_activation, **kwargs))
        else:
            decoder.append(Dense(input_dim,
                                 input_dim=layer_sizes[-1],
                                 activation=output_activation,
                                 **kwargs))
        decoder = containers.Sequential(decoder)

        self.model = Sequential()
        self.model.add(GaussianNoise(noise_sigma, input_shape=(input_dim,)))
        self.model.add(AutoEncoder(encoder, decoder, output_reconstruction=True))
        self.model.compile(loss=loss, optimizer=optimizer)


class TimeConvolutionNetwork(KerasModel):

    def __init__(self):

        super(TimeConvolutionNetwork, self).__init__()

    def create(self):

        pass



