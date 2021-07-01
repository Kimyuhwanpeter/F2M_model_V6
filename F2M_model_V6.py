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
    mean_features, colorization_kernels = tf.nn.moments(features, [1, 2], keepdims=True)
    normalized_features = tf.math.divide(
        tf.subtract(features, mean_features), tf.sqrt(tf.add(colorization_kernels, epsilon)))
    return normalized_features, colorization_kernels, mean_features

def conv(x, channels, kernel=3, stride=1, pad=1, pad_type='zero', scope='conv_0'):

    with tf.compat.v1.variable_scope(scope):
        if pad_type == "zero":
            x = tf.pad(x, [[0, 0],[pad, pad],[pad, pad],[0, 0]])
        if pad_type == "reflect":
            x = tf.pad(x, [[0, 0],[pad, pad],[pad, pad],[0, 0]], mode="REFLECT")

        x = Conv2D(filters=channels, kernel_size=kernel, strides=stride, kernel_regularizer=l2)(x)
        return x

def extract_image_features():
    return tf.keras.applications.VGG19(include_top=False ,input_shape=(256, 256, 3))

def self_attention(x, size, scope='self_attention'):

    with tf.compat.v1.variable_scope(scope):
        C = x.shape[3]
        f = conv(x, C // 2, kernel=1, stride=1, pad=0, scope='f_conv')  # [B, h, w, c']
        g = conv(x, C // 2, kernel=1, stride=1, pad=0, scope='g_conv')  # [B, h, w, c']
        h = conv(x, C, kernel=1, stride=1, pad=0, scope='h_conv')
    
        s = tf.matmul(tf.reshape(g, shape=[g.shape[0], -1, g.shape[-1]]), tf.reshape(f, shape=[f.shape[0], -1, f.shape[-1]]), transpose_b=True)
        # N = h*w, [B, N, N]
        beta = tf.nn.softmax(s)

        o = tf.matmul(beta, tf.reshape(h, shape=[h.shape[0], -1, h.shape[-1]])) # [B, N, C]
        o = tf.reshape(o, shape=size)   # [B, h, w, C]

    return o


def AAMS_generator(input_shape=(256, 256, 3), batch_size=2):  # two inputs

    h_c = inputs_c = tf.keras.Input(input_shape, batch_size=batch_size)
    h_s = inputs_s = tf.keras.Input(input_shape, batch_size=batch_size)

    # freeze
    #####################################################################################################################################
    conv_1_c = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2, name="conv1_c")(h_c)  # 256 x 256 x 64
    conv_2_c = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2, name='conv2_c')(conv_1_c)  # 128 x 128 x 128
    conv_3_c = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2, name='conv3_c')(conv_2_c)  # 64 x 64 x 256
    h_content = Conv2D(filters=512, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2, name='conv4_c')(conv_3_c)  # 32 x 32 x 512
    #####################################################################################################################################

    # train
    #####################################################################################################################################
    conv_1_s = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2, name="conv1_s")(h_s)  # 256 x 256 x 64
    conv_2_s = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2, name='conv2_s')(conv_1_s)  # 128 x 128 x 128
    conv_3_s = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2, name='conv3_s')(conv_2_s)  # 64 x 64 x 256
    h_style = Conv2D(filters=512, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2, name='conv4_s')(conv_3_s)  # 32 x 32 x 512
    #####################################################################################################################################

    projected_hidden_feature, colorization_kernels, mean_features = adain_normalization(h_content)

    attention_feature_map = self_attention(projected_hidden_feature, tf.shape(projected_hidden_feature))

    # 내일은 여기를 이어서 작성해야한다! 기억해!! 오케이!?

    return tf.keras.Model(inputs=[inputs_c, inputs_s], outputs=[h_content, h_style])

def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

mo = AAMS_generator()
mo.get_layer("conv1_c").set_weights(extract_image_features().get_layer('block1_conv1').get_weights())
freeze_all(mo.get_layer("conv1_c"))
mo.get_layer("conv2_c").set_weights(extract_image_features().get_layer('block2_conv1').get_weights())
freeze_all(mo.get_layer("conv2_c"))
mo.get_layer("conv3_c").set_weights(extract_image_features().get_layer('block3_conv1').get_weights())
freeze_all(mo.get_layer("conv3_c"))
mo.get_layer("conv4_c").set_weights(extract_image_features().get_layer('block4_conv1').get_weights())
freeze_all(mo.get_layer("conv4_c"))

mo.get_layer("conv1_s").set_weights(extract_image_features().get_layer('block1_conv1').get_weights())
mo.get_layer("conv2_s").set_weights(extract_image_features().get_layer('block2_conv1').get_weights())
mo.get_layer("conv3_s").set_weights(extract_image_features().get_layer('block3_conv1').get_weights())
mo.get_layer("conv4_s").set_weights(extract_image_features().get_layer('block4_conv1').get_weights())


#h_1 = extract_image_features().get_layer('block1_conv2').output
#h_2 = extract_image_features().get_layer('block2_conv2').output
#h_3 = extract_image_features().get_layer('block3_conv4').output
#h_4 = extract_image_features().get_layer('block4_conv4').output
