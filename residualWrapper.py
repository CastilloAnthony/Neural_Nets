import tensorflow as tf

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model, **kwargs):
    super().__init__(**kwargs)
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each time step is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return inputs + delta

  def getModel(self):
    return self.model

  def save(self, *args, **kwargs):
    return self.model.save(*args, **kwargs)

  def get_config(self):
    base_config = super().get_config()
    config = {
        "model": tf.keras.saving.serialize_keras_object(self.model),
    }
    return {**base_config, **config}

  @classmethod
  def from_config(cls, config):
    sublayer_config = config.pop("model")
    sublayer = tf.keras.saving.deserialize_keras_object(sublayer_config)
    return cls(sublayer, **config)
# end ResidualWrapper