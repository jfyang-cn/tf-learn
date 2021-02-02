from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import inception_v3,mobilenet,vgg16,resnet50 

def classifier(classs_num=5, input_width=224, input_height=224, backbone='vgg16', train_base=False):

    # mobilenet
#     base_model = mobilenet.MobileNet(include_top=False, weights="imagenet", input_tensor=Input(shape=(224,224,3)))
#     x = base_model.output

    if backbone == 'vgg16':
        base_model = vgg16.VGG16(include_top=False, weights="imagenet", input_tensor=Input(shape=(input_width,input_height,3)))
        x = base_model.output
#         if input_width == 224:
#             x = base_model.get_layer('block5_pool').output
#         else:
#             x = base_model.get_layer('block4_conv3').output
    elif backbone == 'resnet50':
        base_model = resnet50.ResNet50(include_top=False, weights="imagenet", input_tensor=Input(shape=(input_width,input_height,3)))
        if input_width == 224:
            x = base_model.output
        else:
            x = base_model.get_layer('activation_21').output
    elif backbone == 'inception_v3':
        base_model = inception_v3.InceptionV3(include_top=False, weights="imagenet", input_tensor=Input(shape=(input_width,input_height,3)))
        x = base_model.output
    else:
        pass

    # My Block 5
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='my_block5_conv1')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='my_block5_conv2')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='my_block5_conv3')(x)
#     x = BatchNormalization()(x,training=False)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='my_block5_pool')(x)    
#     x = Dropout(0.5)(x)

    # InceptionV3
#     base_model = inception_v3.InceptionV3(include_top=False, weights="imagenet", input_tensor=Input(shape=(input_width,input_height,3)))
#     x = base_model.output    
    
    x = GlobalAveragePooling2D()(x)
#     x = Dense(512,activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(512, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(256,activation='relu')(x)
#     x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(classs_num,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = train_base

    return model
