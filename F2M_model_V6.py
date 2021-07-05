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

def extract_image_features():
    return tf.keras.applications.VGG19(include_top=False ,input_shape=(256, 256, 3))

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
    
def conv(x, channels, kernel=3, stride=1, pad=1, pad_type='zero', scope='conv_0'):

    with tf.compat.v1.variable_scope(scope):
        if pad_type == "zero":
            x = tf.pad(x, [[0, 0],[pad, pad],[pad, pad],[0, 0]])
        if pad_type == "reflect":
            x = tf.pad(x, [[0, 0],[pad, pad],[pad, pad],[0, 0]], mode="REFLECT")

        x = Conv2D(filters=channels, kernel_size=kernel, strides=stride, kernel_regularizer=l2)(x)
        return x

def adain_normalization(features):
    epsilon = 1e-7
    mean_features, colorization_kernels = tf.nn.moments(features, [1, 2], keepdims=True)
    normalized_features = tf.math.divide(
        tf.subtract(features, mean_features), tf.sqrt(tf.add(colorization_kernels, epsilon)))
    return normalized_features, colorization_kernels, mean_features

def adain_colorization(normalized_features, colorization_kernels, mean_features):
    return tf.sqrt(colorization_kernels) * normalized_features + mean_features

def adaptive_instance_normalization(content_feature, style_feature):
    normalized_content_feature = InstanceNormalization()(content_feature)
    inst_mean, inst_var = tf.nn.moments(style_feature, [1, 2], keepdims=True)
    return tf.sqrt(inst_var) * normalized_content_feature + inst_mean

def zca_normalization(features):
    # Zero-phase component anlasis (zca)
    shape = tf.shape(features)

    mean_features = tf.reduce_mean(features, axis=[1,2], keepdims=True)
    unbiased_features = tf.reshape(features - mean_features, shape=(shape[0], -1, shape[3]))

    # get 공분산 행렬 (고유벡터 및 값을 알아야하기 떄문, pca와 비슷)
    gram = tf.matmul(unbiased_features, unbiased_features, transpose_a=True)
    gram /= tf.reduce_prod(tf.cast(shape[1:3], tf.float32))

    s, u, v = tf.compat.v1.svd(gram, compute_uv=True)    # 특잇값 분해
    s = tf.expand_dims(s, axis=1)  # let it be active in the last dimension

    # get the effective singular values
    valid_index = tf.cast(s > 0.00001, dtype=tf.float32)
    s_effective = tf.maximum(s, 0.00001)
    sqrt_s_effective = tf.sqrt(s_effective) * valid_index
    sqrt_inv_s_effective = tf.sqrt(1.0/s_effective) * valid_index

    # colorization functions
    colorization_kernel = tf.matmul(tf.multiply(u, sqrt_s_effective), v, transpose_b=True)

    # normalized features
    normalized_features = tf.matmul(unbiased_features, u)
    normalized_features = tf.multiply(normalized_features, sqrt_inv_s_effective)
    normalized_features = tf.matmul(normalized_features, v, transpose_b=True)
    normalized_features = tf.reshape(normalized_features, shape=shape)

    return normalized_features, colorization_kernel, mean_features

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

#@tf.function
def multi_scale_style_swap(contents, styles, patch_size=5):

    c_shape = tf.shape(contents)
    s_shape = tf.shape(styles)  # style의 배치는 1
    # both channels must same !

    c_height, c_width, c_channel = c_shape[1], c_shape[2], c_shape[3]
    proposed_outputs = []
    output = 0.
    for beta in [1.0/2, 1.0/(2**0.5), 1.0]:
        # convert the style features into convolutional kernels
        new_height = tf.cast(tf.multiply(tf.cast(s_shape[1], tf.float32), beta), tf.int32)
        new_width = tf.cast(tf.multiply(tf.cast(s_shape[2], tf.float32), beta), tf.int32)

        tmp_style_features = tf.image.resize(styles, [new_height, new_width])

        style_kernels = tf.compat.v1.extract_image_patches(tmp_style_features, ksizes=[1, patch_size, patch_size, 1],
                                                            strides=[1,1,1,1], rates=[1,1,1,1],
                                                            padding="SAME")  # [1, H, W, p * p * C]
        style_kernels = tf.squeeze(style_kernels, axis=0)  # [H, W, patch_size * patch_size * channel]
        style_kernels = tf.transpose(style_kernels, perm=[2, 0, 1])  # [patch_size * patch_size * channel, H, W]

        # gather the conv and deconv kernels
        deconv_kernels = tf.reshape(
            style_kernels, shape=(patch_size, patch_size, c_channel, -1))
            
        kernels_norm = tf.norm(style_kernels, axis=0, keepdims=True)
        kernels_norm = tf.reshape(kernels_norm, shape=(1, 1, 1, -1))

        # calculate the normalization factor
        mask = tf.ones((c_height, c_width), tf.float32)
        fullmask = tf.zeros((c_height + patch_size - 1, c_width + patch_size - 1), tf.float32)
        for x in range(patch_size):
            for y in range(patch_size):
                paddings = [[x, patch_size - x - 1], [y, patch_size - y - 1]]
                padded_mask = tf.pad(mask, paddings=paddings, mode="CONSTANT")
                fullmask += padded_mask
        pad_width = int((patch_size - 1) / 2)
        deconv_norm = tf.slice(fullmask, [pad_width, pad_width], [c_height, c_width])
        deconv_norm = tf.reshape(deconv_norm, shape=(1, c_height, c_width, 1))

        ########################
        # starting convolution #
        ########################
        # padding operation
        pad_total = patch_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
            
        # convolutional operations
        net = tf.pad(contents, paddings=paddings, mode="REFLECT")
        net = tf.nn.conv2d(
            net,
            tf.compat.v1.div(deconv_kernels, kernels_norm + 1e-7),
            strides=[1, 1, 1, 1],
            padding='VALID')
        # find the maximum locations
        best_match_ids = tf.argmax(net, axis=3)
        best_match_ids = tf.cast(
            tf.one_hot(best_match_ids, depth=tf.shape(net)[3]), dtype=tf.float32)
            
        # find the patches and warping the output
        unnormalized_output = tf.nn.conv2d_transpose(
            input=best_match_ids,
            filters=deconv_kernels,
            output_shape=(c_shape[0], c_height + pad_total, c_width + pad_total, c_channel),
            strides=[1, 1, 1, 1],
            padding='VALID')
        unnormalized_output = tf.slice(unnormalized_output, [0, pad_beg, pad_beg, 0], c_shape)
        output = tf.compat.v1.div(unnormalized_output, deconv_norm)
        output += tf.reshape(output, shape=c_shape)
        #proposed_outputs.append(output)
        
    output /= 3.
    proposed_outputs = output
    proposed_outputs = tf.convert_to_tensor(proposed_outputs, dtype=tf.float32)

    return proposed_outputs

def AAMS_generator(input_shape=(256, 256, 3), batch_size=2):  # two inputs

    h_c = inputs_c = tf.keras.Input(input_shape, batch_size=batch_size) 
    h_s = inputs_s = tf.keras.Input(input_shape, batch_size=batch_size) # style은 오직 배치를 하나씩!

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

    hidden_feature = tf.multiply(projected_hidden_feature, attention_feature_map) + projected_hidden_feature
    hidden_feature = adain_colorization(hidden_feature, colorization_kernels, mean_features)

    # 이 부분은 style branch
    projected_content_features, _, _ = zca_normalization(h_content)
    projected_style_features, style_kernels, mean_style_features = zca_normalization(h_style)

    multi_swapped_features = []
    for i in range(batch_size):
        multi_swapped_features.append(multi_scale_style_swap(tf.expand_dims(projected_content_features[i], 0), tf.expand_dims(projected_style_features[i], 0)))
    multi_swapped_features = tf.convert_to_tensor(multi_swapped_features, dtype=tf.float32)
    multi_swapped_features = tf.squeeze(multi_swapped_features, 1)

    hidden_feature = (hidden_feature + multi_swapped_features) * 0.5

    # decode
    ################################################################################################
    h = conv(hidden_feature, 256, 3, 1, scope="inv_conv_4_1")
    h = ReLU()(h)
    h = tf.keras.layers.UpSampling2D((2,2))(h)
    h = conv(h, 256, 3, 1, scope="inv_conv_3_4")
    h = ReLU()(h)
    h = conv(h, 256, 3, 1, scope="inv_conv_3_3")
    h = ReLU()(h)
    h = conv(h, 256, 3, 1, scope="inv_conv_3_2")
    h = adaptive_instance_normalization(h, conv_3_s)
    h = ReLU()(h)

    h = conv(h, 128, 3, 1, scope="inv_conv_3_1")
    h = ReLU()(h)
    h = tf.keras.layers.UpSampling2D((2,2))(h)
    h = conv(h, 128, 3, 1, scope="inv_conv_2_2")
    h = adaptive_instance_normalization(h, conv_2_s)
    h = ReLU()(h)

    h = conv(h, 64, 3, 1, scope="inv_conv_2_1")
    h = ReLU()(h)
    h = tf.keras.layers.UpSampling2D((2,2))(h)
    h = conv(h, 64, 3, 1, scope="inv_conv_1_2")
    h = adaptive_instance_normalization(h, conv_1_s)
    h = ReLU()(h)

    h = conv(h, 3, 3, 1, scope="inv_conv_1_1") + 127.5      #??


    return tf.keras.Model(inputs=[inputs_c, inputs_s], outputs=h)

def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

mo = AAMS_generator()
mo.summary()
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
