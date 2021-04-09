from tensorflow.keras.callbacks import Callback

class LossMonitor(Callback):
    
#     def on_train_begin(self, logs=None):
#         keys = list(logs.keys())
#         print("Starting training; got log keys: {}".format(keys))
        
    def on_epoch_end(self, epoch, logs={}):
        pass
#         print(
#             "The positive loss for epoch {} is {:7.2f} "
#             "and negative loss is {:7.2f}.".format(
#                 epoch, logs["p_loss"], logs["n_loss"]
#             )
#         )