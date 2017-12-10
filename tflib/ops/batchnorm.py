import tflib as lib

import numpy as np
import tensorflow as tf

def Batchnorm(name, inputs, is_training=None, momentum=0.9, epsilon=2e-5, in_place_update=True):
    if in_place_update:
        result = tf.contrib.layers.batch_norm(input,
                                              decay=momentum,
                                              center=True,
                                              scale=True,
                                              epsilon=epsilon,
                                              updates_collections=None,
                                              is_training=is_training,
                                              scope=name)
        # print(result.get_shape())
        return result
    else:
        result = tf.contrib.layers.batch_norm(input,
                                              decay=momentum,
                                              center=True,
                                              scale=True,
                                              epsilon=epsilon,
                                              is_training=is_training,
                                              scope=name)
        # print(result.get_shape())
        return result
