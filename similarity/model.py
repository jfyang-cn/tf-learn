from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPooling2D,BatchNormalization,Activation,ReLU,Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import inception_v3,mobilenet,vgg16,resnet50,MobileNetV2
from layer import TripleLoss


def similarity_model(input_width=224, input_height=224, backbone='vgg16', train_base=False):
    
    base_model = None
    predictions = None
    model_input = None
    if backbone == 'vgg16':
        base_model = vgg16.VGG16(include_top=True, weights="imagenet", input_tensor=Input(shape=(input_width,input_height,3)))
#         x = base_model.get_layer('block5_pool').output
        x = base_model.output
        model_input = base_model.input
    elif backbone == 'resnet50':
        base_model = resnet50.ResNet50(include_top=True, weights="imagenet", input_tensor=Input(shape=(input_width,input_height,3)))
        x = base_model.output
        model_input = base_model.input
    elif backbone == 'inception_v3':
        base_model = inception_v3.InceptionV3(include_top=True, weights="imagenet", input_tensor=Input(shape=(input_width,input_height,3)))
        model_input = base_model.input
        x = base_model.get_layer('avg_pool').output
#         x = Dense(1024)(x)
#         x = BatchNormalization()(x, training=True)
#         x = ReLU(max_value=1.0)(x)
#         x = Dropout(0.5)(x)
#         x = Dense(1024)(x)
#         x = BatchNormalization()(x, training=False)
#         x = Activation('relu', max_value=1.0)(x)
#         x = Dropout(0.5)(x)
#         x = Dense(256)(x)
#         x = BatchNormalization()(x, training=False)
#         x = ReLU(max_value=1.0)(x)
#         x = Dropout(0.5)(x)
#         x = Dense(128)(x)
#         x = BatchNormalization()(x, training=False)
#         x = Activation('relu', max_value=1.0)(x)
#         x = Dropout(0.5)(x)
    elif backbone == 'mobilenet_v2':
        base_model = MobileNetV2(include_top=True, weights="imagenet", input_tensor=Input(shape=(input_width,input_height,3)))
#         x = base_model.output
        x = base_model.get_layer('Conv_1_bn').output
        x = GlobalAveragePooling2D()(x)
#         x = base_model.get_layer('global_average_pooling2d').output
        model_input = base_model.input
    else:
        x = Input(shape=(input_width,input_height,3))
        model_input = x
        x = Conv2D(32, (3,3), strides=(2, 2), padding='valid', use_bias=False)(x)
#         x = BatchNormalization(scale=False)(x)
#         x = ReLU(max_value=1.0)(x)
        x = GlobalAveragePooling2D()(x)
        
#         loss_func = TripleLoss(32, name='loss_layer')
        
#         x = loss_func(x)
    
    predictions = x
    model = Model(inputs=model_input,outputs=predictions)    
    
    if base_model is not None:
        istrain = train_base
        for i,layer in enumerate(base_model.layers):
#             if layer.name == 'mixed8':
#                 print('mixed8 ', i, ' enable training')
#                 istrain = True

            layer.trainable = istrain

    return model