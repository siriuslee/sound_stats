import h5py
import logging
import copy
import os
import numpy as np
import theano
from keras.models import Sequential, model_from_json
from keras.layers.core import (Dense, TimeDistributedDense, Dropout,
                               ActivityRegularization, Activation)
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import SimpleRNN
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

class KerasModel(Sequential):

    def __init__(self, output_directory=None, **kwargs):

        self.output_directory = output_directory
        super(KerasModel, self).__init__(**kwargs)
        if self.output_directory is not None:
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)

    def fit(self, X, y=None, batch_size=None, shuffle="batch",
            callbacks=None, validation_split=0.1, es_epochs=5, checkpoint=True,
            **kwargs):

        if batch_size is None:
            if hasattr(X, "chunks"):
                batch_size = X.chunks[0]
            else:
                batch_size = 128

        if callbacks is None:
            callbacks = list()
        callbacks.append(EarlyStopping(patience=es_epochs))
        if checkpoint and self.output_directory is not None:
            filepath = os.path.join(self.output_directory,
                                    "weights.{epoch:02d}.hdf5")
            callbacks.append(ModelCheckpoint(filepath, verbose=1))

        if y is None:
            y = X

        return super(KerasModel, self).fit(X, y,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           validation_split=validation_split,
                                           callbacks=callbacks,
                                           **kwargs)

    def get_layer_output(self, X, layer):

        get_feature = theano.function([self.layers[0].input],
                                      layer.get_output(train=False),
                                      allow_input_downcast=True)

        return [out for out in get_feature(X)]

    def get_filters(self, layer, input_shape=None):

        layers = list()
        while True:
            layers.insert(0, layer)
            if hasattr(layer, "previous"):
                layer = layer.previous
            else:
                break

        layer_num = 0
        for layer in reversed(layers):
            if hasattr(layer, "W"):
                layer_num += 1
                if layer_num == 1:
                    filters = layer.get_weights()[0]
                else:
                    filters = np.dot(filters, layer.get_weights()[0])

        if input_shape is None:
            input_shape = filters[:, ii].shape

        return [filters[:, ii].reshape(input_shape) for ii in range(filters.shape[1])]

    def evaluate(self, X, y=None, **kwargs):

        if y is None:
            y = X

        return super(KerasModel, self).evaluate(X, y, **kwargs)

    def compile(self, loss="mean_squared_error", optimizer="adam", **kwargs):

        return super(KerasModel, self).compile(loss=loss,
                                               optimizer=optimizer,
                                               **kwargs)

    def compute_statistics(self, X=None, layer=None, outputs=None):

        if outputs is None:
            outputs = self.get_output(X, layer)
        outputs = np.vstack(outputs)
        self.covariance = np.cov(outputs, rowvar=False)
        self.mean = np.mean(outputs, axis=0)

        return self.mean, self.covariance

    @classmethod
    def load(cls, directory):

        weights_file = os.path.join(directory, "weights.h5")
        model_file = os.path.join(directory, "model.json")

        with open(model_file, "r") as f:
            json_string = f.read()

        model = model_from_json(json_string)
        model.load_weights(weights_file)
        model.__class__ = cls

        return model

    def save(self, directory=None, overwrite=False, weights=True,
             model_file="models.json", weights_file="weights.h5"):

        if directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
        else:
            directory = self.output_directory

        weights_file = os.path.join(directory, weights_file)
        model_file = os.path.join(directory, model_file)
        if overwrite or not os.path.isfile(model_file):
            with open(model_file, "w") as f:
                f.write(self.to_json())
            if weights:
                self.save_weights(weights_file, overwrite=overwrite)

    def get_config(self, *args, **kwargs):

        config = super(KerasModel, self).get_config(*args, **kwargs)
        config["name"] = "Sequential"

        return config

    @staticmethod
    def _format_params_nlayers(param, nlayers):

        if isinstance(param, (list, tuple)):
            if len(param) != nlayers:
                raise ValueError("If parameter is a list, it must be the same length as the number of layers: %d" % nlayers)
        else:
            param = [param] * nlayers

        return param


class Convolution1DAutoEncoder(KerasModel):

    def __init__(self, input_dim,
                 layer_sizes,
                 input_filter_length,
                 noise_sigma=0.1,
                 activation="relu",
                 batch_normalization=True,
                 dropout=0.5,
                 output_activation="linear",
                 output_dim=None,
                 output_filter_length=None,
                 output_directory=None,
                 **kwargs):

        super(Convolution1DAutoEncoder, self).__init__(output_directory,
                                                       **kwargs)

        # Construct the model
        nlayers = len(layer_sizes)

        # Check length of parameter lists
        activation = self._format_params_nlayers(activation, nlayers)
        dropout = self._format_params_nlayers(dropout, nlayers)

        # Output parameters defaults to those of the previous layers
        if output_dim is None:
            output_dim = input_dim
        if output_filter_length is None:
            output_filter_length = input_filter_length

        # Add input noise - need input_shape...
        # if noise_sigma > 0:
        #     self.add(GaussianNoise(noise_sigma, input_dim=input_dim))

        # Create all layers
        n_units, layer_sizes = layer_sizes[0], layer_sizes[1:]
        input_layer = Convolution1D(n_units,
                                    input_filter_length,
                                    border_mode="same",
                                    input_dim=input_dim,
                                    **kwargs)
        output_layer = Convolution1D(output_dim,
                                     output_filter_length,
                                     border_mode="same",
                                     **kwargs)
        layers = [input_layer]
        for n_units in layer_sizes:
            hidden_layer = TimeDistributedDense(n_units, **kwargs)
            layers.append(hidden_layer)

        for ii, layer in enumerate(layers):
            self.add(layer)
            if batch_normalization:
                self.add(BatchNormalization())
            self.add(Activation(activation[ii]))
            if dropout[ii] > 0:
                self.add(Dropout(dropout[ii]))
        self.add(output_layer)
        self.add(Activation(output_activation))

        if self.output_directory is not None:
            self.save(model_file="init_model.json", weights=False)


class RecurrentAutoEncoder(KerasModel):

    def __init__(self, input_dim,
                 layer_sizes,
                 noise_sigma=0,
                 activation="tanh",
                 batch_normalization=False,
                 dropout=0,
                 output_activation=None,
                 output_dim=None,
                 output_directory=None,
                 **kwargs):


        super(RecurrentAutoEncoder, self).__init__(output_directory,
                                                   **kwargs)

        # Construct the model
        nlayers = len(layer_sizes)

        # Check length of parameter lists
        activation = self._format_params_nlayers(activation, nlayers)
        dropout = self._format_params_nlayers(dropout, nlayers)

        if isinstance(input_dim, tuple):
            input_shape = input_dim
            input_dim = input_shape[0]
        else:
            input_shape = None

        # Output parameters defaults to those of the previous layers
        if output_dim is None:
            output_dim = input_dim

        # Add input noise - need input_shape...
        if noise_sigma > 0:
            if input_shape is not None:
                self.add(GaussianNoise(noise_sigma, input_shape=input_shape))
            else:
                raise ValueError("If you want to add noise, make sure input_dim is a tuple of the expected input shape")

        # Create all layers
        n_units, layer_sizes = layer_sizes[0], layer_sizes[1:]
        input_layer = SimpleRNN(n_units,
                                return_sequences=True,
                                input_dim=input_dim,
                                **kwargs)
        output_layer = TimeDistributedDense(output_dim,
                                            **kwargs)
        layers = [input_layer]
        for n_units in layer_sizes:
            hidden_layer = SimpleRNN(n_units,
                                     return_sequences=True,
                                     **kwargs)
            layers.append(hidden_layer)

        for ii, layer in enumerate(layers):
            self.add(layer)
            if batch_normalization:
                self.add(BatchNormalization())
            self.add(Activation(activation[ii]))
            if dropout[ii] > 0:
                self.add(Dropout(dropout[ii]))
        self.add(output_layer)
        self.add(Activation(output_activation))

        if self.output_directory is not None:
            self.save(model_file="init_model.json", weights=False)


class StaticAutoEncoder(KerasModel):

    _save_attrs = ["input_dim"]

    def __init__(self, **kwargs):

        super(StaticAutoEncoder, self).__init__(**kwargs)

    def create(self, input_dim,
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

        self.input_dim = input_dim
        if isinstance(activation, list):
            if len(activation) != len(layer_sizes):
                raise ValueError("If activation is a list it must be of the same length as layer sizes")
        else:
            activation = [activation] * len(layer_sizes)

        input_dim = np.prod(self.input_dim)

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

    def get_encoder_layer(self, layer):

        ae = [ll for ll in self.model.layers if isinstance(ll, AutoEncoder)][0]
        layers = [ll for ll in ae.encoder.layers if hasattr(ll, "W")]

        try:
            return layers[layer]
        except IndexError:
            raise IndexError("Model only has %d encoder layers with weights" % len(layers))

    def get_filters(self, layer_num, input_shape=None):

        for ii in range(layer_num + 1):
            ll = self.get_encoder_layer(ii)
            if ii == 0:
                filters = ll.get_weights()[0]
            else:
                filters = np.dot(filters, ll.get_weights()[0])
        if input_shape is None:
            input_shape = filters[:, ii].shape

        return [filters[:, ii].reshape(input_shape) for ii in range(filters.shape[1])]

    def sample(self, layer_num, mean=None, covariance=None, nsamples=1):

        if mean is None:
            mean = self.mean
        if covariance is None:
            covariance = self.covariance

        ae = [ll for ll in self.model.layers if isinstance(ll, AutoEncoder)][0]
        input_layer = ae.encoder.layers[layer_num]
        layers = ae.encoder.layers[layer_num + 1:] + ae.decoder.layers
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
