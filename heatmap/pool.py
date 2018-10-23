from keras import backend as K

def min_max_pool2d(x):
    max_x = K.pool2d(x, pool_size=(7, 7), strides=(2, 2), data_format="channels_first")#dim_ordering="th")
    min_x = -K.pool2d(-x, pool_size=(7, 7), strides=(2, 2), data_format="channels_first")#dim_ordering="th")
    return K.concatenate([max_x, min_x], axis =1)  # concatenate on channel
#data_format="channels_first")
def min_max_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] *= 2
    shape[2] = 1
    shape[3] = 1
    return tuple(shape)
