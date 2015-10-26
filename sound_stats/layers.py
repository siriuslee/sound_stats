from keras.layers.core import MaskedLayer
from keras.layers.convolutional import Convolution2D, conv_output_length


class TimeConvolution(Convolution2D):
    input_ndim = 3

    def __init__(self, nb_filter, nb_row, nb_delay, **kwargs):

        border_mode = kwargs.pop("border_mode", "full")
        if border_mode != "full":
            raise ValueError("Border mode must be full")

        self.nb_delay = nb_delay
        super(TimeConvolution, self).__init__(nb_filter, nb_row, nb_delay, border_mode=border_mode, **kwargs)

    @property
    def output_shape(self):

        input_shape = self.input_shape
        cols = input_shape[3]
        cols = conv_output_length(cols, self.nb_col, self.border_mode, self.subsample[1])
        return (input_shape[0], self.nb_filter, cols)

    def get_output(self, train=False):

        output = super(TimeConvolution, self).get_output(train=train)
        valid = int(output.shape[3] / 2)

        return output[:, :, valid, :]

    def get_config(self):

        config = {"name": self.__class__.__name__,
                  "nb_delay": self.nb_delay}
        base_config = super(Convolution2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class TimePad(MaskedLayer):

    def __init__(self, left_pad, right_pad, **kwargs):

        super(TimePad, self).__init__(**kwargs)
        self.left_pad = left_pad
        self.right_pad = right_pad

    def get_output(self, train=False):

        X = self.get_input(train)
        nsamples, stack_size, nrows, ncols = X.shape
        if self.left_pad > 0:
            X = np.hstack([np.zeros((nsamples, stack_size, nrows, self.left_pad)), X])
        if self.right_pad > 0:
            X = np.hstack([X, np.zeros((nsamples, stack_size, nrows, self.right_pad))])

        return X