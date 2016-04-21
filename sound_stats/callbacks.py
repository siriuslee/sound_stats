import os
import random
import numpy as np
import matplotlib
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
        for dl in self.decoding_layers:
            dl.trainable = False

    def on_batch_end(self, batch, logs={}):

        for dl, el in zip(self.decoding_layers, reversed(self.encoding_layers)):

            W = el.get_weights()[0]
            b = dl.get_weights()[1]
            dl.set_weights((W.T, b))


class PlotInputOutput(Callback):
    """
    Plots the input and output of the network for a few training or validation examples
    """

    def __init__(self, inputs, nepochs=1, nsamples=5,
                 output_directory=None, transpose=True):

        inds = random.sample(range(inputs.shape[0]), nsamples)
        self.inputs = [inputs[ii][:] for ii in inds]
        self.nepochs = nepochs
        self.nsamples = nsamples
        self.output_directory = output_directory
        self.transpose = transpose
        self.figs = list()
        self.axs = list()

        if self.output_directory is None:
            plt.ion()
        else:
            plt.switch_backend("Agg")
            if not os.path.isdir(self.output_directory):
                raise IOError("Could not find directory %s" % self.output_directory)

    def get_filename(self, epoch, example):
        name = ["example", "%d" % example, "epoch", "%d" % epoch]
        return "_".join(name) + ".png"

    def on_train_begin(self, logs={}):

        self.plot(0)

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.nepochs == 0:
            self.plot(epoch)

    def plot(self, epoch):

        outputs = [output for output in self.model.predict(np.array(self.inputs))]
        inputs = self.inputs
        if self.transpose:
            inputs = map(np.transpose, inputs)
            outputs = map(np.transpose, outputs)

        if len(self.figs) == 0:
            for ii in range(self.nsamples):
                fig, axs = plt.subplots(2, 1)
                self.figs.append(fig)
                self.axs.append(axs)

        for ii, (fig, axs) in enumerate(zip(self.figs, self.axs)):
            fig.suptitle("%d) Epoch %d" % (ii, epoch))
            input = inputs[ii]
            output = outputs[ii]
            if len(input.shape) == 1:
                axs[0].plot(input)
            else:
                axs[0].imshow(input, origin="lower", aspect="auto",
                              interpolation="none")
            if len(output.shape) == 1:
                axs[1].plot(output)
            else:
                axs[1].imshow(output, origin="lower", aspect="auto",
                              interpolation="none")

        if self.output_directory is None:
            plt.draw()
            plt.pause(0.01)
            plt.show()
        else:
            for ii, fig in enumerate(self.figs):
                filename = self.get_filename(epoch, ii)
                fig.savefig(os.path.join(self.output_directory, filename),
                            dpi=300,
                            facecolor="white",
                            edgecolor="white")


class PlotFilters(Callback):
    """
    Plots up to 20 random filters from the specified layer.
    """

    def __init__(self, layer, nfilters=20, nepochs=1,
                 filter_shape=None, output_directory=None):

        super(PlotFilters, self).__init__()

        self.layer = layer
        self.nepochs = nepochs
        self.nfilters = nfilters
        if filter_shape is None:
            self.filter_shape = self.layer.W_shape
        else:
            self.filter_shape = filter_shape
        self.output_directory = output_directory

        self.filter_inds = None
        self.fig = None
        self.axs = None

        if self.output_directory is None:
            plt.ion()
        else:
            if not os.path.isdir(self.output_directory):
                raise IOError("Could not find directory %s" % self.output_directory)

    def get_filename(self, epoch):
        name = ["filters"]
        if self.layer.name is not None:
            name.append(self.layer.name)
        name.append("epoch")
        name.append("%d" % epoch)

        return "_".join(name) + ".png"

    def on_train_begin(self, logs={}):

        self.filter_inds = random.sample(range(self.layer.output_dim),
                                         self.nfilters)
        self.plot(-1)

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.nepochs == 0:
            self.plot(epoch)

    def plot(self, epoch):

        filters = self.model.get_filters(self.layer,
                                         input_shape=self.filter_shape)
        filters = [filters[ii] for ii in self.filter_inds]

        self.nfilters = len(filters)

        nrows = int(np.sqrt(self.nfilters))
        ncols = int(float(self.nfilters - 1) / nrows) + 1
        if self.fig is None:
            self.fig, self.axs = plt.subplots(nrows, ncols)
        if self.nfilters <= 1:
            self.axs = [self.axs]

        self.fig.suptitle("Epoch %d" % epoch)
        for ax, filt in zip(self.axs.ravel(), filters):
            if len(filt.squeeze().shape) == len(filt.shape):
                ax.plot(filt)
            else:
                ax.imshow(filt, origin="lower", aspect="auto", interpolation="none")

        if self.output_directory is None:
            plt.draw()
            plt.pause(0.01)
            plt.show()
        else:
            filename = self.get_filename(epoch)
            fig.savefig(os.path.join(self.output_directory, filename),
                        dpi=300,
                        facecolor="white",
                        edgecolor="white")
