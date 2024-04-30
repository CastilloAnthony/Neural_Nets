### Refer to this page: https://www.tensorflow.org/tutorials/structured_data/time_series ###
import tensorflow as tf

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index
    # end __init__

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    # end call
# end Baseline