"""
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# class Ancilayer(tf.keras.layers.Layer):      #暂时先不按层的派生类组织该结构，改用函数组织
#     def __init__(self, **kwargs):
#         super(Ancilayer, self).__init__()
#     def call(self,x):
#         return self.Ancilayer_op()
#     def Ancilayer_op(self):
#         pass
def Compositeblock(x,units,height,width):
    # block 的高和宽暂时写死 有需要后面改
    conv = layers.Conv2D(units, (height, width), padding='SAME', activation="relu")   # 似乎应该使用leakyrelu
    batchnorm = layers.BatchNormalization()
    y = batchnorm(conv(x))
    return  y

def Anciblock(x):
    y0 = Compositeblock(x,16,7,7)
    y1 = Compositeblock(y0,16,5,5)
    y2 = Compositeblock(y0,16,3,3)
    y = y1 + y2
    return y

def Denoiser(x_in):
    temp = Compositeblock(x_in,64,7,7)            #命名以后方便抽出来做 two_stage training
    temp = Compositeblock(temp,16,1,1)
    for i in range(4):
        temp = temp + Anciblock(temp)

    output = layers.Conv2D(2,(3,3),padding="same",activation="sigmoid",name="denoiser_output")(temp) #命名以后方便抽出来做 two_stage training
    return output

def Encoder(x,feedback_bits):
    B=4
    with tf.compat.v1.variable_scope('Encoder'):
        x_d = Denoiser(x)                 # denoiser 预处理
        x = Compositeblock(x_d,64,7,7)
        x = Compositeblock(x, 16, 1, 1)

        x += Anciblock(x)

        x = Compositeblock(x,2,1,1)
        x = layers.Flatten()(x)                      # 和reshape等价
        x = layers.Dense(units=int(feedback_bits / B), activation='sigmoid')(x)
        encoder_output = QuantizationLayer(B)(x)
    return encoder_output

def Decoder(x,feedback_bits):
    B=4
    decoder_input = DeuantizationLayer(B)(x)
    x = tf.keras.layers.Reshape((-1, int(feedback_bits/B)))(decoder_input)
    x = layers.Dense(1024, activation='sigmoid')(x)
    x_ini = layers.Reshape((16, 32, 2))(x)
    x = Compositeblock(x_ini,16,1,1)
    for i in range(3):
        x += Anciblock(x)

    decoder_output = layers.Conv2D(2, 3, padding = 'SAME',activation="sigmoid")(x_ini)
    return decoder_output



# Next parts are all referred baseline
#This part realizes the quantization and dequantization operations.
#The output of the encoder must be the bitstream.

def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, 4:]).reshape(-1,
                                                                                                            Num_.shape[
                                                                                                                1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)

    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)

    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)

    def custom_grad(dy):
        grad = dy
        return (grad, grad)

    return result, custom_grad


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config

@tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)

    return result, custom_grad


class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


# offered by baseline
# def Encoder(x,feedback_bits):
#     B=4
#     with tf.compat.v1.variable_scope('Encoder'):
#         x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
#         x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
#         x = layers.Flatten()(x)
#         x = layers.Dense(units=int(feedback_bits/B), activation='sigmoid')(x)
#         encoder_output = QuantizationLayer(B)(x)
#     return encoder_output
# def Decoder(x,feedback_bits):
#     B=4
#     decoder_input = DeuantizationLayer(B)(x)
#     x = tf.keras.layers.Reshape((-1, int(feedback_bits/B)))(decoder_input)
#     x = layers.Dense(1024, activation='sigmoid')(x)
#     x_ini = layers.Reshape((16, 32, 2))(x)
#
#     for i in range(3):
#         x = layers.Conv2D(8, 3, padding = 'SAME',activation="relu")(x_ini)
#         x = layers.Conv2D(16,3, padding = 'SAME',activation="relu")(x)
#         x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
#         x_ini = keras.layers.Add()([x_ini, x])
#
#
#     decoder_output = layers.Conv2D(2, 3, padding = 'SAME',activation="sigmoid")(x_ini)
#
#     return decoder_output


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def Score(NMSE):
    score = 1-NMSE
    return score

# Return keywords of your own custom layers to ensure that model
# can be successfully loaded in test file.
def get_custom_objects():
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer}
