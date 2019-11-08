
import tensorflow as tf


class SqueezeLayer(tf.keras.layers.Layer):
    """ a class that squeezes a given axis of a tensor"""
    
    def __init__(self):
        super(SqueezeLayer, self).__init__()

    def call(self, input_tensor, axis=3):
        try:
            output = tf.squeeze(input_tensor, axis)
        except:
            output = input_tensor
        return output

class DotLayer(tf.keras.layers.Layer):
    """ Return cosine similarity between two columns of two matrices. """

    def __init__(self):
        super(DotLayer, self).__init__()

    def call(self, list_tensors):
        layer = tf.keras.layers.Dot(axes=[2, 2], normalize= True)
        output_dot = layer(list_tensors)
        output_diag = tf.matrix_diag_part(output_dot)
        return output_diag


# our defined loss function based on binary cross entropy
def loss_BCE_custom(cos_scores_sig):
    """
    Return binary cross entropy loss for cosine similarity layer.

    :param cos_scores_sig: array of float numbers, output of the cosine similarity
        layer followed by sigmoid function.
    :return: a function, which will be used as a loss function in model.compile.
    """

    def loss(y_true, y_pred):
        part_pos = tf.keras.backend.sum(-y_true * tf.keras.backend.log(cos_scores_sig), axis= -1)
        part_neg = tf.keras.backend.sum((y_true-1)*tf.keras.backend.log(1-cos_scores_sig), axis= -1)
        return (part_pos + part_neg) / tf.keras.backend.int_shape(cos_scores_sig)[-1]

    return loss


# the proposed LSTM model
def lstm_model(shape_eeg, shape_env, units_lstm=16, filters_cnn_eeg=8, filters_cnn_env=16, units_hidden=20,
                           stride_temporal=3, kerSize_ver_eeg=7, kerSize_temporal=9,
                           stride_ch=2, fun_act='tanh'):
    """
    Return a LSTM based model where batch normalization is applied to input of each layer.

    :param shape_eeg: a numpy array, shape of EEG signal (time, channel, 1)
    :param shape_env: a numpy array, shape of envelope signal (time, 1, 1)
    :param units_lstm: an int, number of units in LSTM
    :param filters_cnn_eeg: an int, number of CNN filters applied on EEG
    :param filters_cnn_env: an int, number of CNN filters applied on envelope
    :param units_hidden: an int, number of units in the first time_distributed layer
    :param stride_temporal: an int, amount of stride in the temporal direction
    :param kerSize_ver_eeg: an int, size of CNN filter kernel in the direction of EEG channels
    :param kerSize_temporal: an int, size of CNN filter kernel in the temporal direction
    :param stride_ch: an int, amount of stride in the channel direction for CNN
    :param fun_act: activation function used in layers
    :return: LSTM based model
    """

    ############
    #### upper part of network dealing with EEG.
    input_eeg = tf.keras.layers.Input(shape=shape_eeg)
    layer_exp1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=3))
    input_eeg1 = layer_exp1(input_eeg)
    input_eeg_BN = tf.keras.layers.BatchNormalization()(input_eeg1)       # batch normalization

    output1_eeg = tf.keras.layers.Convolution2D(filters_cnn_eeg, (kerSize_temporal,kerSize_ver_eeg),
                                                strides=(stride_temporal, stride_ch), activation=fun_act)(input_eeg_BN)
    output1_eeg = tf.keras.layers.BatchNormalization()(output1_eeg)      # batch normalization

    # output1_eeg = SqueezeLayer()(output1_eeg, axis=2)
    layer_permute = tf.keras.layers.Permute((1, 3, 2))
    output1_eeg = layer_permute(output1_eeg)  # size = (210,4,8)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output1_eeg)[1],
                                             tf.keras.backend.int_shape(output1_eeg)[2] *
                                             tf.keras.backend.int_shape(output1_eeg)[3]))
    output1_eeg = layer_reshape(output1_eeg)  # size = (210,32)

    layer2_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_hidden, activation = fun_act))
    output2_eeg = layer2_timeDis(output1_eeg)
    output2_eeg = tf.keras.layers.BatchNormalization()(output2_eeg)

    layer3_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_lstm, activation = fun_act))
    output3_eeg = layer3_timeDis(output2_eeg) #size = (210,16)

    ##############
    #### Bottom part of the network dealing with Envelope.
    input_env = tf.keras.layers.Input(shape=shape_env)
    input_non_used_env = tf.keras.layers.Input(shape=shape_env)
    input_env1 = layer_exp1(input_env)
    input_non_used_env1 = layer_exp1(input_non_used_env)

    input_env_BN = tf.keras.layers.BatchNormalization()(input_env1)

    output1_mel = tf.keras.layers.Convolution2D(filters_cnn_env, (kerSize_temporal, 1),
                                                strides=(stride_temporal, 1), activation= fun_act)(input_env_BN)
    output1_mel = tf.keras.layers.BatchNormalization()(output1_mel)

    layer_permute = tf.keras.layers.Permute((1,3,2))
    output1_mel = layer_permute(output1_mel)  # size = (210,4,8)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output1_mel)[1],
                                             tf.keras.backend.int_shape(output1_mel)[2]*tf.keras.backend.int_shape(output1_mel)[3]))
    output1_mel = layer_reshape(output1_mel)    # size = (210,32)

    lstm_mel = tf.keras.layers.LSTM(units_lstm, return_sequences=True, activation= fun_act)
    output2_mel = lstm_mel(output1_mel)  # size = (210,16)

    ##############
    #### last common layers
    layer_dot = DotLayer()
    cos_scores = layer_dot([output3_eeg, output2_mel])

    layer_expand = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=2))
    layer_sigmoid = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    cos_scores_sig = layer_sigmoid(layer_expand(cos_scores))    # shape = (batch,212,1)

    layer_ave = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))
    cos_scores_sig = SqueezeLayer()(cos_scores_sig, axis=2)
    y_pred = layer_ave(cos_scores_sig)

    model = tf.keras.Model(inputs=[input_eeg, input_env, input_non_used_env], outputs=y_pred)

    return model, cos_scores, cos_scores_sig
