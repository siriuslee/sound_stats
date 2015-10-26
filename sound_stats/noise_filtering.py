import h5py
import numpy as np
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D

def filters_to_weights(filename):

    tmp = loadmat(filename, variable_names=["strf"])
    filters = [filt[0] for filt in tmp["strf"]["filter"]]

    nfilters = len(filters)
    nfreq, ndelays = filters[0].shape
    weights = [np.array(filters).reshape((nfilters, 1, nfreq, ndelays)),
               np.zeros((nfilters,))]

    return weights

### NOTES:
### If the reconstruction delays range from a0 to aM and the encoding delays range from b0 to bM, then the total number of time points needed for a single batch is (aM + bM) - (a0 + b0) + (batch_size - 1). Traditionally this has been (0 + 99) - (-99 + 0) + (batch_size - 1) = 197 + batch_size.




class NoiseFilteringModel(object):

    def create(self, input_shape, layer_sizes=None, weights=None, encoding_delays=100,
               optimizer="sgd", loss="mean_squared_error"):
        """
        Creates a noise filtering model
        :param input_shape: Size of the mini-batch spectrograms (F x T)
        :param layer_sizes: Size of each layer (default [100, 80, 100])
        :params weights: List of initialization weights for each layer. Can also use a string to specify a keras initialization (default glorot_uniform).
        :params encoding_delays: The delays for the encoding filters
        :params decoding_delays: The delays for the decoding filters
        :param optimizer: Instance from keras.optimizers or string referring to one (default "sgd")
        :param loss: String referring to keras objective function
        :return: a compiled keras model instance
        """

        if layer_sizes is None:
            layer_sizes = [100, 80, 100]
        nlayers = len(layer_sizes)

        if weights is None:
            weights = ["glorot_uniform"] * nlayers
        elif isinstance(weights, str):
            weights = [weights] * nlayers

        # Create the encoding layers
        encoding_layers = layer_sizes[:int(nlayers / 2) + 1]
        encoder = Sequential()
        for ll in xrange(len(encoding_layers)):
            layer_kwargs = dict()
            if isinstance(weights[ll], str):
                layer_kwargs["init"] = weights[ll]
            else:
                layer_kwargs["weights"] = weights[ll]

            if ll == 0:
                # Layer 0, add an input layer
                layer = Convolution2D(layer_sizes[ll], nfreq, ndelays, input_shape=(1, input_shape[0], input_shape[1]), **layer_kwargs)
            else:
                encoder.add(Dense(layer_sizes[ll - 1], layer_sizes[ll]))

            model.add(layer)
            encoder.add(Activation("relu")) # Or some activation function
            # maybe add dropout here as well

        decoding_layers = layer_sizes[int(nlayers / 2):]
        decoder = Sequential()
        for ll in xrange(1, len(layer_sizes) + 1):
            if ll == 0:
                decoder.add(Dense(layer_sizes[-ll], input_size[0] * input_size[1]))
            else:
                decoder.add(Dense(layer_sizes[-(ll + 1)], layer_sizes[-ll]))
            decoder.add(Activation('relu'))

        model = Sequential()
        model.add(Autoencoder(encoder, decoder, output_reconstruction=False))