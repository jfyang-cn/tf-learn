from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

def c3d(class_num,input_width,input_height,input_depth,backbone,train_base):

    input_shape = (input_depth,input_width,input_height,3)
    weight_decay = 0.005
    
    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((1,2,2),strides=(1,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num,kernel_regularizer=l2(weight_decay),name='top_%s' % (class_num))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    
    for i,layer in enumerate(model.layers):
        if i <= 15:
            layer.trainable = train_base
    
    return model
