from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import time
import numpy as np

def timestampe():
    return int(time.time())

class TripleLoss(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(TripleLoss, self).__init__(**kwargs)
        
#         self.loss_log = {}

    def build(self, input_shape):
        # 添加可训练参数
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer='glorot_normal',
#                                       trainable=True)
#         self.bias = self.add_weight(name='bias',
#                                     shape=(self.output_dim,),
#                                     initializer='zeros',
#                                     trainable=True)
#         self.centers = self.add_weight(name='centers',
#                                        shape=(self.output_dim, input_shape[1]),
#                                        initializer='glorot_normal',
#                                        trainable=True)
        pass

    def call(self, inputs):
        # 对于center loss来说，返回结果还是跟Dense的返回结果一致
        # 所以还是普通的矩阵乘法加上偏置
        self.inputs = inputs
        return inputs #K.dot(inputs, self.kernel) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def loss(self, y_true, y_pred):
        # 定义完整的loss
#         y_true = K.cast(y_true, 'int32') # 保证y_true的dtype为int32
#         crossentropy = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
#         centers = K.gather(self.centers, y_true[:, 0]) # 取出样本中心
#         center_loss = K.sum(K.square(centers - self.inputs), axis=1) # 计算center loss
#         return crossentropy + lamb * center_loss
        a = K.expand_dims(y_pred[0], axis=0)
        a = K.tile(a, [16,1])
        p = y_pred[0:16]
        n = y_pred[16:32]
        margin = 1.0

        p_dis = K.mean(K.square(a-p), axis=-1)
        n_dis = K.mean(K.square(a-n), axis=-1)
        
        loss_val = p_dis-n_dis+margin
        
#         self.loss_log[timestampe()] = str(p_dis)+','+str(n_dis)

        return loss_val
    
    def save(self):
        pass
#         print(self.loss_log)
#         np.save('loss_dict.npy', self.loss_log)
