import numpy as np
import h5py
import tensorflow as tf
from Model_define_tf import NMSE, Score, get_custom_objects
from sklearn import model_selection


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
x_train, x_test = model_selection.train_test_split(data, test_size=0.05, random_state=1)

# load encoder_output
decode_input = np.load('./Modelsave/encoder_output.npy')

# load model and test NMSE
decoder_address = './Modelsave/decoder.h5'
_custom_objects = get_custom_objects()  # load keywords of Custom layers
model_decoder = tf.keras.models.load_model(decoder_address, custom_objects=_custom_objects)
y_test = model_decoder.predict(decode_input)
print('The NMSE is ' + np.str(NMSE(x_test, y_test)))

