# -*- coding:utf-8 -*-
import tensorflow as tf

l2 = tf.keras.regularizers.l2(0.000005)
Conv2D = tf.keras.layers.Conv2D
TransConv2D = tf.keras.layers.Conv2DTranspose
DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
Maxpool2D = tf.keras.layers.MaxPool2D
ZeroPadd2D = tf.keras.layers.ZeroPadding2D
ReLU = tf.keras.layers.ReLU
LeakReLU = tf.keras.layers.LeakyReLU
# https://openaccess.thecvf.com/content_CVPR_2019/papers/Yao_Attention-Aware_Multi-Stroke_Style_Transfer_CVPR_2019_paper.pdf
# https://github.com/JianqiangRen/AAMS/blob/master/net/aams.py
class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
    
def adain_normalization(features):
    epsilon = 1e-7
    mean_features, colorization_kernels = tf.nn.moments(features, [1, 2], keep_dims=True)
    normalized_features = tf.math.divide(
        tf.subtract(features, mean_features), tf.sqrt(tf.add(colorization_kernels, epsilon)))
    return normalized_features, colorization_kernels, mean_features

def conv(x, channels, kernel=3, stride=1, pad=1, pad_type='zero', scope='conv_0'):

    with tf.compat.v1.variable_scope(scope):
        if pad_type == "zero":
            x = tf.pad(x, [[0, 0],[pad, pad],[pad, pad],[0, 0]])
        if pad_type == "reflect":
            x = tf.pad(x, [[0, 0],[pad, pad],[pad, pad],[0, 0]], mode="REFLECT")

        x = Conv2D(filters=channels, kernel_size=kernel, strides=stride, kernel_regularizer=l2)
        return x

def extract_image_features():
    return tf.keras.applications.VGG19(include_top=False ,input_shape=(256, 256, 3))

# ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4']


def AAMS_generator(input_shape=(256, 256, 3)):  # two inputs

    h = inputs = tf.keras.Input(input_shape)

    h = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2, name="conv1")  # 256 x 256 x 64
    
    h = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2, name='conv2')(h)  # 128 x 128 x 128

    h = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2, name='conv3')(h)  # 64 x 64 x 256

    h = Conv2D(filters=512, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2, name='conv4')(h)  # 32 x 32 x 512

    projected_hidden_feature, colorization_kernels, mean_features = adain_normalization(h)



    return tf.keras.Model(inputs=inputs, outputs=h)

mo = AAMS_generator()
mo.get_layer("conv1").set_weights(extract_image_features().get_layer('block1_conv1').get_weights())
mo.get_layer("conv2").set_weights(extract_image_features().get_layer('block2_conv1').get_weights())
mo.get_layer("conv3").set_weights(extract_image_features().get_layer('block3_conv1').get_weights())
mo.get_layer("conv4").set_weights(extract_image_features().get_layer('block4_conv1').get_weights())

#h_1 = extract_image_features().get_layer('block1_conv2').output
#h_2 = extract_image_features().get_layer('block2_conv2').output
#h_3 = extract_image_features().get_layer('block3_conv4').output
#h_4 = extract_image_features().get_layer('block4_conv4').output
