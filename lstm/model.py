import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Input, CuDNNLSTM, LSTM, GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate, Dense, Dropout, Conv2D, Flatten, LSTMCell, RNN, Lambda
# LSTM, CuDNNLSTM are not quantized
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from mobilenet import get_mobilenet_base

n_neurons = 64

# Custom Layer cann't be quantized
class MyLSTM(Layer):
    # Input shape
    #     (None,9,1024)
    #
    # 

    def compute_output_shape(self, input_shape):
        return (None, 9, n_neurons)

    def call(self, inputs):

        out = tf.unstack(out, axis=1)
        cell = tf.nn.rnn_cell.LSTMCell(n_neurons)
        out, state  = tf.nn.static_rnn(cell, out, dtype=tf.float32)
        out = tf.stack(out)                      # (9, None, 64)
        out = tf.transpose(out, perm=[1, 0, 2])  # (None, 9, 64)

        return out
    
    def get_config(self):
        base_config = super(MyLSTM, self).get_config()
        return dict(list(base_config.items()))

def s_model_vgg(input_shape):
    
    img_input = layers.Input(shape=input_shape)
    
        # Block 1
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s1_conv1')(img_input)
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='s1_pool')(x)

    # Block 2
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s2_conv1')(x)
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='s2_pool')(x)

    # Block 3
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s3_conv1')(x)
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s3_conv2')(x)
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='s3_pool')(x)

    # Block 4
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s4_conv1')(x)
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s4_conv2')(x)
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='s4_pool')(x)

    # Block 5
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s5_conv1')(x)
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s5_conv2')(x)
    x = layers.Conv2D(8, (3, 3),
                      activation='relu',
                      padding='same',
                      name='s5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='s5_pool')(x)
    
    #
    x = layers.Flatten(name='s6_flatten')(x)
    x = layers.Reshape((7*7, 8), name='s6_reshape')(x)
    x = layers.Permute((2,1), name='s6_permute')(x)

    inputs = img_input
    model = Model(inputs, x, name='s_model')

    return model

def create_model_pretrain0(n_output, is_training=True):
    
    base_1 = get_mobilenet_base(input_shape=(224,224,3), with_weights=True)
    out_1 = base_1.output
#     out_1 = GlobalAveragePooling2D()(out_1)            # (None, 1024)
#     out_1 = layers.Reshape((1, 1024), name='block15_reshape')(out_1)

    # merge two stream
#     out_ = concatenate([out_1, out_2], axis=1)           # (None, 9, 1024) 
    out_ = Flatten()(out_1)

    out_ = Dense(n_output, activation='softmax', name='s_out')(out_)
    
    if is_training is not True:
        out_ = tf.maximum(out_, -1e27, name='s_max')

    model = Model(inputs=[base_1.input], outputs=[out_])
    
    return model

def lstm_func(x):
    cell = layers.LSTMCell(n_neurons)
#     cell = layers.Lambda(lambda t: tf.nn.rnn_cell.LSTMCell(n_neurons))
#     out = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)
    x = layers.Lambda(lambda t: tf.nn.static_rnn(cell, t, dtype=tf.float32))(x)
    return x

# based on mobilenet v1
def create_model_pretrain(n_output, is_training=True):
    # input_1 = Input(shape=(224, 224, 3), name='input_1')
    # input_2 = Input(shape=(n_neurons, 224, 224), name='input_2')

#     base_1 = MobileNetV2(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
#     out_1 = base_1.get_layer('block_15_add').output    # (None, 7, 7, 160)

    # merge input, cause rknn does not support different channel number for multiple inputs
    input_ = Input(shape=(224, 224, 3+8), name='input_')
    input_1 = Lambda(lambda x: x[:,:,:,0:3])(input_)
    input_2 = Lambda(lambda x: x[:,:,:,3:8+3])(input_)

    base_1 = get_mobilenet_base(input_1, input_shape=(224,224,3), with_weights=True)
    out_1 = base_1.output
    out_1 = GlobalAveragePooling2D()(out_1)            # (None, 1024)
    out_1 = layers.Reshape((1, 1024), name='block15_reshape')(out_1)

#     base_2 = s_model(input_shape=(224, 224, 8))
#     out_2 = base_2.output                              # (None, 8, 49)
#     out_2 = LSTM(n_neurons, name='s_lstm')(out_2)      # CuDNNLSTM is alternative 

    base_2 = get_mobilenet_base(input_2, input_shape=(224,224,8), with_weights=False)
    out_2 = base_2.output
#     out_2 = layers.Flatten(name='s6_flatten')(out_2)
#     out_2 = layers.Permute((3,1,2), name='s6_permute')(out_2)      # rknn does not support this op
    out_2 = layers.Reshape((8, 1024), name='s6_reshape')(out_2)
#     out_2 = layers.Permute((2,1), name='s6_permute')(out_2)
#     out_2 = GlobalAveragePooling2D()(out_2)            # (None, 160)

    # merge two stream
    out_ = concatenate([out_1, out_2], axis=1)           # (None, 9, 1024) 
#     out_ = LSTM(n_neurons, return_sequences=False, name='s_lstm')(out_)    # CuDNNLSTM is alternative 
#     out_ = Flatten()(out_)

# quantize error
    cell = LSTMCell(n_neurons, name='lstm_cell')
    out_ = RNN(cell=cell, unroll=True, name='rnn')(out_)

# quantize error
#     out_ = layers.SimpleRNN(n_neurons)(out_)

# quantize error
#     out_ = MyLSTM()(out_)
#     out_ = Flatten()(out_)

# lstm_cell takes an input argument in call
#     out_ = layers.Lambda(lambda t: tf.unstack(t, axis=1))(out_)
#     out_ = layers.Lambda(lambda t: lstm_func(t))(out_)
#     out_ = layers.Lambda(lambda t: tf.stack(t))(out_)
#     out_ = layers.Lambda(lambda t: tf.transpose(t, perm=[1, 0, 2]))(out_)
#     out_ = Flatten()(out_)

#     out_ = Dense(64, activation='relu')(out_)
#     out_ = Dropout(.5)(out_)
#     out_ = Dense(32, activation='relu')(out_)
#     out_ = Dropout(.5)(out_)
    out_ = Dense(n_output, activation='softmax', name='s_out')(out_)
    
    if is_training is not True:
        out_ = tf.maximum(out_, -1e27, name='s_max')

    model = Model(inputs=[input_], outputs=[out_])
#     model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['acc'])
#     model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['acc'])
#     model.compile(RMSprop(learning_rate=0.001, rho=0.9), loss='sparse_categorical_crossentropy', metrics=['acc'])
#     model.summary()
    
    return model

# based on mobilenet v2
def create_model_pretrain2(n_output, is_training=True):
    # input_1 = Input(shape=(224, 224, 3), name='input_1')
    # input_2 = Input(shape=(n_neurons, 224, 224), name='input_2')

    base_1 = MobileNetV2(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
    out_1 = base_1.get_layer('block_15_add').output    # (None, 7, 7, 160)
    out_1 = GlobalAveragePooling2D()(out_1)            # (None, 160)

    base_2 = s_model_vgg(input_shape=(224, 224, 8))
    out_2 = base_2.output                              # (None, 8, 49)
    out_2 = LSTM(n_neurons, return_sequences=False, name='s_lstm')(out_2)      # (None, 8)

    # merge two stream
    out_ = concatenate([out_1, out_2])
    out_ = Dense(64, activation='relu')(out_)
    out_ = Dropout(.5)(out_)
    out_ = Dense(32, activation='relu')(out_)
    out_ = Dropout(.5)(out_)
    out_ = Dense(n_output, activation='softmax')(out_)

    model = Model(inputs=[base_1.input, base_2.input], outputs=out_)
#     model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['acc'])
    model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['acc'])
#     model.compile(RMSprop(learning_rate=0.001, rho=0.9), loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.summary()
    
    return model

# based on mobilenet v1
def create_model(n_output, is_training=True):
    input_1 = Input(shape=(224, 224, 3), name='input_1')
    input_2 = Input(shape=(224, 224, 8), name='input_2')

#     base_1 = MobileNetV2(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
#     out_1 = base_1.get_layer('block_15_add').output    # (None, 7, 7, 160)

    # merge input, cause rknn does not support different channel number for multiple inputs
#     input_ = Input(shape=(224, 224, 3+8), name='input_')
#     input_1 = Lambda(lambda x: x[:,:,:,0:3])(input_)
#     input_2 = Lambda(lambda x: x[:,:,:,3:8+3])(input_)

    base_1 = get_mobilenet_base(input_1, input_shape=(224,224,3), with_weights=True)
    out_1 = base_1.output
    out_1 = GlobalAveragePooling2D()(out_1)              # (None, 1024)
    out_1 = layers.Reshape((1, 1024), name='block15_reshape')(out_1)

#     base_2 = s_model(input_shape=(224, 224, 8))
#     out_2 = base_2.output                              # (None, 8, 49)
#     out_2 = LSTM(n_neurons, name='s_lstm')(out_2)      # CuDNNLSTM is alternative 

    base_2 = get_mobilenet_base(input_2, input_shape=(224,224,8), with_weights=False)
    out_2 = base_2.output
#     out_2 = layers.Flatten(name='s6_flatten')(out_2)
    out_2 = layers.Permute((3,1,2), name='s6_permute')(out_2)      # rknn does not support this op
    out_2 = layers.Reshape((8, 1024), name='s6_reshape')(out_2)
#     out_2 = layers.Permute((2,1), name='s6_permute')(out_2)
#     out_2 = GlobalAveragePooling2D()(out_2)            # (None, 160)

    # merge two stream
    out_ = concatenate([out_1, out_2], axis=1)           # (None, 9, 1024) 
#     out_ = LSTM(n_neurons, return_sequences=False, name='s_lstm')(out_)    # CuDNNLSTM is alternative 
#     out_ = Flatten()(out_)

# quantize error
    cell = LSTMCell(n_neurons, name='lstm_cell')
    out_ = RNN(cell=cell, unroll=True, name='rnn')(out_)

# quantize error
#     out_ = layers.SimpleRNN(n_neurons)(out_)

# quantize error
#     out_ = MyLSTM()(out_)
#     out_ = Flatten()(out_)

# lstm_cell takes an input argument in call
#     out_ = layers.Lambda(lambda t: tf.unstack(t, axis=1))(out_)
#     out_ = layers.Lambda(lambda t: lstm_func(t))(out_)
#     out_ = layers.Lambda(lambda t: tf.stack(t))(out_)
#     out_ = layers.Lambda(lambda t: tf.transpose(t, perm=[1, 0, 2]))(out_)
#     out_ = Flatten()(out_)

    out_ = Dense(64, activation='relu')(out_)
    out_ = Dropout(.5)(out_)
    out_ = Dense(32, activation='relu')(out_)
    out_ = Dropout(.5)(out_)
    out_ = Dense(n_output, activation='softmax', name='s_out')(out_)
    
    if is_training is not True:
        out_ = tf.maximum(out_, -1e27, name='s_max')

    model = Model(inputs=[input_1, input_2], outputs=[out_])
#     model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['acc'])
#     model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['acc'])
#     model.compile(RMSprop(learning_rate=0.001, rho=0.9), loss='sparse_categorical_crossentropy', metrics=['acc'])
#     model.summary()
    
    return model

# create_model_pretrain(6)