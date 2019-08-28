import os
from time import gmtime, strftime

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint,TensorBoard


def make_tensorboard(log_root_dir="logs",set_dir_name=" "):
    tictoc = strftime("%a_%d_%b_%Y_%M_%S", gmtime())
    directory_name = tictoc
    log_dir = set_dir_name + "_" + directory_name
    log_dir = os.path.join(log_root_dir,log_dir)
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir)
    return tensorboard


def early_stopping():
    return EarlyStopping(monitor="val_loss", min_delta=0, patience=0, verbose=0, mode="auto")


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get("loss"))


def loss_history():
    return LossHistory()

def model_checkpoint(model_dir="model"):
    return ModelCheckpoint(filepath=os.path.join(model_dir,"model-{epoch:02d}.h5"))
