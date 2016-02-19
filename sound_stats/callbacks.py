import random
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.layers.core import AutoEncoder


class ConstrainAutoEncoder(Callback):
    """
    Constrains the weights of the decoding layers to be the transpose of the corresponding encoding layers.
    """

    def on_train_begin(self, logs={}):

        ae = [ll for ll in self.model.layers if isinstance(ll, AutoEncoder)][0]
        self.encoding_layers = [ll for ll in ae.encoder.layers if hasattr(ll, "W")]
        self.decoding_layers = [ll for ll in ae.decoder.layers if hasattr(ll, "W")]

    def on_batch_end(self, batch, logs={}):

        for dl, el in zip(self.decoding_layers, reversed(self.encoding_layers)):

            W = el.get_weights()[0]
            b = dl.get_weights()[1]
            dl.set_weights((W.T, b))


class ZeroBias(Callback):
    """
    Constrains all layers to have 0 bias
    """

    def on_train_begin(self, logs={}):

        ae = [ll for ll in self.model.layers if isinstance(ll, AutoEncoder)][0]
        encoding_layers = [ll for ll in ae.encoder.layers if hasattr(ll, "W")]
        decoding_layers = [ll for ll in ae.decoder.layers if hasattr(ll, "W")]
        self.layers = encoding_layers + decoding_layers

    def on_batch_end(self, batch, logs={}):

        for ll in self.layers:
            W, b = ll.get_weights()
            ll.set_weights((W, np.zeros_like(b)))


class PlotFilters(Callback):
    """
    Plots up to 20 random filters from the specified layer.
    """

    def __init__(self, network, layer_num=0, nepochs=1):

        super(PlotFilters, self).__init__()

        self.network = network
        self.layer_num = layer_num
        self.nepochs = nepochs

        self.filter_inds = None
        self.fig = None
        self.axs = None

        plt.ion()

    def on_train_begin(self, logs={}):

        layer = self.network.get_encoder_layer(self.layer_num)
        n_filters = min(layer.output_dim, 20)
        self.filter_inds = random.sample(range(layer.output_dim), n_filters)
        self.plot("Begin training")

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.nepochs == 0:
            self.plot("Epoch %d" % epoch)

    def plot(self, title=""):

        filters = self.network.get_filters(self.layer_num)
        filters = [filters[ii] for ii in self.filter_inds]
        n_filters = len(filters)

        nrows = int(np.sqrt(n_filters))
        ncols = int(float(n_filters - 1) / nrows) + 1
        self.fig, self.axs = plt.subplots(nrows, ncols)
        if n_filters <= 1:
            self.axs = [self.axs]

        self.fig.suptitle(title)
        for ax, filt in zip(self.axs, filters):
            ax.imshow(filt, origin="lower", aspect="auto", interpolation="none")

        plt.show()



