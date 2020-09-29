import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import get_source_inputs

# conv1 (Conv2D)               (None, None, None, 32)    864       
# _________________________________________________________________
# conv1_bn (BatchNormalization (None, None, None, 32)    128       
# _________________________________________________________________
# conv1_relu (ReLU)            (None, None, None, 32)    0         
# _________________________________________________________________
# conv_dw_1 (DepthwiseConv2D)  (None, None, None, 32)    288       
# _________________________________________________________________
# conv_dw_1_bn (BatchNormaliza (None, None, None, 32)    128       
# _________________________________________________________________
# conv_dw_1_relu (ReLU)        (None, None, None, 32)    0         
# _________________________________________________________________

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_pad')(inputs)
    x = tf.keras.layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(fused=True, axis=channel_axis, name='conv1_bn')(x)
    return tf.keras.layers.ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = tf.keras.layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = tf.keras.layers.BatchNormalization(fused=True,
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = tf.keras.layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = tf.keras.layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = tf.keras.layers.BatchNormalization(fused=True,
        axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return tf.keras.layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
    

def _conv_block2(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='s_conv1_pad')(inputs)
    x = tf.keras.layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='s_conv1')(x)
    x = tf.keras.layers.BatchNormalization(fused=True, axis=channel_axis, name='s_conv1_bn')(x)
    return tf.keras.layers.ReLU(6., name='s_conv1_relu')(x)


def _depthwise_conv_block2(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)),
                                 name='s_conv_pad_%d' % block_id)(inputs)
    x = tf.keras.layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='s_conv_dw_%d' % block_id)(x)
    x = tf.keras.layers.BatchNormalization(fused=True,
        axis=channel_axis, name='s_conv_dw_%d_bn' % block_id)(x)
    x = tf.keras.layers.ReLU(6., name='s_conv_dw_%d_relu' % block_id)(x)

    x = tf.keras.layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='s_conv_pw_%d' % block_id)(x)
    x = tf.keras.layers.BatchNormalization(fused=True,
        axis=channel_axis, name='s_conv_pw_%d_bn' % block_id)(x)
    return tf.keras.layers.ReLU(6., name='s_conv_pw_%d_relu' % block_id)(x)
    
def get_mobilenet_base(input_tensor=None, input_shape=(257,257,3), with_weights=False):

    alpha=1.0
    depth_multiplier = 1
    if input_tensor is None:
        input_tensor = tf.keras.layers.Input(shape=input_shape)
    input_tensor_source = get_source_inputs(input_tensor)[0]
    
    if with_weights:
        x = _conv_block(input_tensor, 32, alpha, strides=(2, 2))
        
        x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

        x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=2)
        x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

        x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=4)
        x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=6)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

        x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=12)
        x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    
    else:
        x = _conv_block2(input_tensor, 32, alpha, strides=(2, 2))

        x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier, block_id=1)

        x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=2)
        x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier, block_id=3)

        x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=4)
        x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier, block_id=5)

#         x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier,
#                                   strides=(2, 2), block_id=6)
#         x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier, block_id=7)
#         x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier, block_id=8)
#         x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier, block_id=9)
#         x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier, block_id=10)
#         x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier, block_id=11)
        x = tf.keras.layers.ZeroPadding2D(((2, 2), (2, 2)),
                                 name='s_conv_pad_%d' % 6)(x)
#         x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier,
#                                   strides=(2, 2), block_id=12)
#         x = _depthwise_conv_block2(x, 8, alpha, depth_multiplier, block_id=13)

    model = tf.keras.models.Model(input_tensor_source, x)
    
    if with_weights:
        weights_path = '/home/yourname/.keras/models/mobilenet_1_0_224_tf_no_top.h5'
        model.load_weights(weights_path, by_name=True)

    return model