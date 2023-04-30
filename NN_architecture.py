from keras.layers import *
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from util import leaky_relu

def build_basic_model(in_channel):
    """basic model for Coupling
    """
    _in = Input(shape=(None, None, in_channel))
    _ = _in
    hidden_dim = 512
    get_custom_objects().update({'leaky_relu': Activation(leaky_relu)})
    _ = Conv2D(hidden_dim,
               (3, 3),
               padding='same')(_)
    # _ = Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('leaky_relu')(_)
    _ = Conv2D(hidden_dim,
               (1, 1),
               padding='same')(_)
    # _ = Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('leaky_relu')(_)
    _ = Conv2D(in_channel,
               (3, 3),
               kernel_initializer='zeros',
               padding='same')(_)
    return Model(_in, _)


