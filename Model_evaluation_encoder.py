import numpy as np
import h5py
import tensorflow as tf
import sklearn
from sklearn import model_selection
from Model_define_tf import get_custom_objects

# parameter setting
img_height = 16
img_width = 32
img_channels = 2
feedback_bits = 128
# Data loading
data_load_address = './data'
mat = h5py.File(data_load_address+'/Hdata.mat', 'r')
data = np.transpose(mat['H_train'])
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
data = np.transpose(data, (0, 2, 3, 1))   # change to data_form: 'channel_last'
# data partitioning
x_train, x_test = sklearn.model_selection.train_test_split(data, test_size=0.05, random_state=1)

# load model
encoder_address = './Modelsave/encoder.h5'
_custom_objects = get_custom_objects()  # load keywords of Custom layers
model_encoder = tf.keras.models.load_model(encoder_address, custom_objects=_custom_objects)
encode_feature = model_encoder.predict(x_test)
print("feedbackbits length is ", np.shape(encode_feature)[-1])
np.save('./Modelsave/encoder_output.npy', encode_feature)
