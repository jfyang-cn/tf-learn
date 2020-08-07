from tensorflow.keras.layers import Dense, Input,  Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3, mobilenet,vgg16,resnet50

def classifier(classs_num=5, input_width=224, input_height=224):

    # mobilenet
#     base_model = mobilenet.MobileNet(include_top=False, weights="imagenet", input_tensor=Input(shape=(224,224,3)))
#     x = base_model.output

    # VGG
    base_model = vgg16.VGG16(include_top=False, weights="imagenet", input_tensor=Input(shape=(input_width,input_height,3)))
#     x = base_model.output
    x = base_model.get_layer('block5_conv3').output

    # ResNet
#     base_model = resnet50.ResNet50(include_top=False, weights="imagenet", input_tensor=Input(shape=(input_width,input_height,3)))
#     x = base_model.output
    
    x = GlobalAveragePooling2D()(x)
#     x = Dense(1024,activation='relu')(x)
#     x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(classs_num,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False

    return model

def temp_classifier():
    
    base_model = mobilenet.MobileNet(include_top=False, weights="imagenet", input_tensor=Input(shape=(299,299,3)))
#     print(base_model.input)
    x = base_model.get_layer('conv_pad_12')
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
# #     x = Dense(64,activation='relu')(x)
# #     x = Dropout(0.5)(x)
#     x = Dense(64, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     predictions = Dense(2,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=x.output)
    
#     for layer in base_model.layers:
#         layer.trainable = False

    return model
