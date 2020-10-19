import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import model_selection
import sklearn
from Model_define_tf import Encoder, Decoder, NMSE

import time
import os
from Logger import *

# std_out重定向
sys.stdout = Logger("test.txt", sys.stdout)
sys.stderr = Logger("test.txt", sys.stderr)

print(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime()))
# parameters
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2

# Model construction
# encoder model
Encoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="encoder_input")
Encoder_output = Encoder(Encoder_input, feedback_bits)
encoder = keras.Model(inputs=Encoder_input, outputs=Encoder_output, name='encoder')

# decoder model
Decoder_input = keras.Input(shape=(feedback_bits,), name='decoder_input')
Decoder_output = Decoder(Decoder_input, feedback_bits)
decoder = keras.Model(inputs=Decoder_input, outputs=Decoder_output, name="decoder")

# autoencoder model
autoencoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="original_img")
encoder_out = encoder(autoencoder_input)
decoder_out = decoder(encoder_out)
autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Data loading
data_load_address = './data'
mat = h5py.File(data_load_address+'/Hdata.mat', 'r')
data = np.transpose(mat['H_train'])      # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
data = np.transpose(data, (0, 2, 3, 1))   # change to data_form: 'channel_last'
x_train, x_test = sklearn.model_selection.train_test_split(data, test_size=0.05, random_state=1)


# model saving preparation
start_time = strtime = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime())

# dirpath = os.getcwd()+"/check_point/model_"+start_time
dirpath = "check_point/model_" + start_time
os.makedirs(dirpath)

filepath= dirpath + "/model_{epoch:02d}.h5"

checkpointer=keras.callbacks.ModelCheckpoint(filepath, verbose=1,
 								save_best_only=False,
 								save_weights_only=True, mode='auto',
 								period=1)



# model training
try:
	loadpath = "model_2020_10_13_15_35_24/model_34.h5"
	encoder_address = "check_point/" + loadpath
	autoencoder.load_weights(encoder_address)
	print("load model success")
	history=autoencoder.fit(x=x_train, y=x_train, batch_size=1024+320, epochs=500, validation_split=0.2, callbacks=[checkpointer])  #338

except:
	print("load model failed!!!")
	history=autoencoder.fit(x=x_train, y=x_train, batch_size=512, epochs=1000, validation_split=0.1,callbacks=[checkpointer])

# model save
# save encoder
modelsave1 = './Modelsave/encoder.h5'
encoder.save(modelsave1)
# save decoder
modelsave2 = './Modelsave/decoder.h5'
decoder.save(modelsave2)

# model test
y_test = autoencoder.predict(x_test, batch_size=512)
print('The NMSE is ' + np.str(NMSE(x_test, y_test)))






