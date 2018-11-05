from keras.models import Model
from keras import initializers
from keras import constraints
from keras import regularizers
from keras.layers import *
from recurrentshop.engine import RNNCell
import keras.backend as K

def _slice(x, dim, index):
    return x[:, index * dim: dim * (index + 1)]


def get_slices(x, n):
    dim = int(K.int_shape(x)[1] / n)
    return [Lambda(_slice, arguments={'dim': dim, 'index': i}, output_shape=lambda s: (s[0], dim))(x) for i in range(n)]


class Identity(Layer):

    def call(self, x):
        return x + 0.


class ExtendedRNNCell(RNNCell):

    def __init__(self, units=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 #kernel_initializer='zeros',
                 recurrent_initializer='orthogonal',
                 #recurrent_initializer='zeros',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if units is None:
            assert 'output_dim' in kwargs, 'Missing argument: units'
        else:
            kwargs['output_dim'] = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super(ExtendedRNNCell, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ExtendedRNNCell, self).get_config()
        config.update(base_config)
        return config


class LSTMCell1(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm1 = Input(batch_shape=output_shape)
        kernel = Dense(output_dim * 4,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(output_dim * 4,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=False)
        kernel_out = kernel(x)
        recurrent_kernel_out = recurrent_kernel(h_tm1)
        x0, x1, x2, x3 = get_slices(kernel_out, 4)
        r0, r1, r2, r3 = get_slices(recurrent_kernel_out, 4)
        f = add([x0, r0])
        f = Activation(self.recurrent_activation)(f)
        i = add([x1, r1])
        i = Activation(self.recurrent_activation)(i)
        c_prime = add([x2, r2])
        c_prime = Activation(self.activation)(c_prime)
        c = add([multiply([f, c_tm1]), multiply([i, c_prime])])
        c = Activation(self.activation)(c)
        o = add([x3, r3])
        h = multiply([o, c])
        return Model([x, h_tm1, c_tm1], [h, Identity()(h), c])

class LSTMCell2(RNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm1 = Input(batch_shape=output_shape)
        f = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        f = Activation('sigmoid')(f)
        i = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        i = Activation('sigmoid')(i)
        c_prime = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        c_prime = Activation('tanh')(c_prime)
        c = add([multiply([f, c_tm1]), multiply([i, c_prime])])
        c = Activation('tanh')(c)
        o = add([Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1)])
        h = multiply([o, c])
        return Model([x, h_tm1, c_tm1], [h, h, c])

#version 1, left
class LSTMCell_Left(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm1 = Input(batch_shape=output_shape)
        kernel = Dense(output_dim * 4,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(output_dim * 4,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=False)
        kernel_out = kernel(x)
        recurrent_kernel_out = recurrent_kernel(h_tm1)
        x0, x1, x2, x3 = get_slices(kernel_out, 4)
        r0, r1, r2, r3 = get_slices(recurrent_kernel_out, 4)
        f = add([x0, r0])
        f = Activation(self.recurrent_activation)(f)
        i = add([x1, r1])
        i = Activation(self.recurrent_activation)(i)
        c_prime = add([x2, r2])
        c_prime = Activation(self.activation)(c_prime)
        c = add([multiply([f, c_tm1]), multiply([i, c_prime])])
        c = Activation(self.activation)(c)
        o = add([x3, r3])
        h = multiply([o, c])
        y=concatenate([h, Identity()(h), c])
        return Model([x, h_tm1, c_tm1], [h, Identity()(h), c])

#version 1, right
class TransferLSTM(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        xx = Input(batch_shape=input_shape)
        h_tm1x = Input(batch_shape=output_shape)
        c_tm1x = Input(batch_shape=output_shape)
        x_lx=Lambda(lambda x: x[:,:45])(xx)
        x_h=Lambda(lambda x: x[:,45:90])(xx)
        x_c=Lambda(lambda x: x[:,90:135])(xx)
        x_rx=Lambda(lambda x: x[:,135:])(xx)

        x_kernel = Dense(output_dim*4,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        h_kernel = Dense(output_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        c_kernel = Dense(output_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)


        x=x_kernel(concatenate([x_lx,x_rx]))
        h_tm1=h_kernel(concatenate([h_tm1x,x_h]))
        c_tm1=c_kernel(concatenate([c_tm1x,x_c]))

        '''x=concatenate([x_lx,x_rx])
        #x=concatenate([x,x])
        h_tm1=concatenate([h_tm1x,x_h])
        c_tm1=concatenate([c_tm1x,x_c])'''

        kernel = Dense(output_dim * 4,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(output_dim * 4,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=False)
        kernel_out = kernel(x)
        recurrent_kernel_out = recurrent_kernel(h_tm1)
        x0, x1, x2, x3 = get_slices(kernel_out, 4)
        r0, r1, r2, r3 = get_slices(recurrent_kernel_out, 4)
        f = add([x0, r0])
        f = Activation(self.recurrent_activation)(f)
        i = add([x1, r1])
        i = Activation(self.recurrent_activation)(i)
        c_prime = add([x2, r2])
        c_prime = Activation(self.activation)(c_prime)
        c = add([multiply([f, c_tm1]), multiply([i, c_prime])])
        c = Activation(self.activation)(c)
        o = add([x3, r3])
        h = multiply([o, c])
        return Model([xx, h_tm1x, c_tm1x], [h, Identity()(h), c])

class KerasLSTMCell(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm1 = Input(batch_shape=output_shape)
        kernel = Dense(output_dim * 4,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(output_dim * 4,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=True)

        kernel_out = kernel(x)
        recurrent_kernel_out = recurrent_kernel(h_tm1)
        kernel_i, kernel_f, kernel_c, kernel_o = get_slices(kernel_out, 4)
        recurrent_kernel_i, recurrent_kernel_f, recurrent_kernel_c, recurrent_kernel_o = get_slices(recurrent_kernel_out, 4)

        inputs_i = x
        inputs_f = x
        inputs_c = x
        inputs_o = x

        x_i = multiply([inputs_i, kernel_i])
        x_f = multiply([inputs_f, kernel_f])
        x_c = multiply([inputs_c, kernel_c])
        x_o = multiply([inputs_o, kernel_o])
        if False:#self.use_bias:
            self.bias_i = self.bias[:45]
            self.bias_f = self.bias[45: 45 * 2]
            self.bias_c = self.bias[45 * 2: 45 * 3]
            self.bias_o = self.bias[45 * 3:]

            x_i = add([x_i, self.bias_i])
            x_f = add([x_f, self.bias_f])
            x_c = add([x_c, self.bias_c])
            x_o = add([x_o, self.bias_o])

        if False:#0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        i = Activation(self.recurrent_activation)(add([x_i , multiply([h_tm1_i,
                                                    recurrent_kernel_i])]))
        f = Activation(self.recurrent_activation)(add([x_f , multiply([h_tm1_f,
                                                    recurrent_kernel_f])]))
        temp= Activation(self.activation)(add([x_c , multiply([h_tm1_c,
                                                        recurrent_kernel_c])]))
        c = add([multiply([f, c_tm1]) , multiply([ i ,temp ]) ])
        o = Activation(self.recurrent_activation)(add([x_o , multiply([h_tm1_o,
                                                    recurrent_kernel_o])]))
        temp2=Activation(self.activation)(c)
        h = multiply([o , temp2])

        #y=concatenate([h, Identity()(h), c])
        y=h
        return Model([x, h_tm1, c_tm1], [y, Identity()(h), c])

class KerasLSTMCell_left(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm1 = Input(batch_shape=output_shape)
        kernel = Dense(input_dim * 4,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(input_dim * 4,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=True)

        kernel_out = kernel(x)
        recurrent_kernel_out = recurrent_kernel(h_tm1)
        kernel_i, kernel_f, kernel_c, kernel_o = get_slices(kernel_out, 4)
        recurrent_kernel_i, recurrent_kernel_f, recurrent_kernel_c, recurrent_kernel_o = get_slices(recurrent_kernel_out, 4)

        inputs_i = x
        inputs_f = x
        inputs_c = x
        inputs_o = x


        x_i = kernel_i
        x_f = kernel_f
        x_c = kernel_c
        x_o = kernel_o

        if False:#self.use_bias:
            self.bias_i = self.bias[:45]
            self.bias_f = self.bias[45: 45 * 2]
            self.bias_c = self.bias[45 * 2: 45 * 3]
            self.bias_o = self.bias[45 * 3:]

            x_i = add([x_i, self.bias_i])
            x_f = add([x_f, self.bias_f])
            x_c = add([x_c, self.bias_c])
            x_o = add([x_o, self.bias_o])

        if False:#0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        i = Activation(self.recurrent_activation)(add([x_i ,recurrent_kernel_i]))
        f = Activation(self.recurrent_activation)(add([x_f ,recurrent_kernel_f]))
        temp= Activation(self.activation)(add([x_c, recurrent_kernel_c]))
        c = add([multiply([f, c_tm1]) , multiply([ i ,temp ]) ])
        o = Activation(self.recurrent_activation)(add([x_o , recurrent_kernel_o]))
        temp2=Activation(self.activation)(c)
        h = multiply([o , temp2])

        y=concatenate([h, h_tm1, c_tm1])
       
        return Model([x, h_tm1, c_tm1], [y, Identity()(h), c])

class KerasLSTMCell_right(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)

        xx = Input(batch_shape=input_shape)
        h_tm1x = Input(batch_shape=output_shape)
        c_tm1x = Input(batch_shape=output_shape)
        x_lx=Lambda(lambda x: x[:,:45])(xx)
        x_h=Lambda(lambda x: x[:,45:90])(xx)
        x_c=Lambda(lambda x: x[:,90:135])(xx)
        x_rx=Lambda(lambda x: x[:,135:])(xx)

        # x_kernel = Dense(output_dim,
        #         kernel_initializer=self.kernel_initializer,
        #         kernel_regularizer=self.kernel_regularizer,
        #         kernel_constraint=self.kernel_constraint,
        #         use_bias=self.use_bias,
        #         bias_initializer=self.bias_initializer,
        #         bias_regularizer=self.bias_regularizer,
        #         bias_constraint=self.bias_constraint,
			# 	activation=self.recurrent_activation)
        # h_kernel = Dense(output_dim,
        #         kernel_initializer=self.kernel_initializer,
        #         kernel_regularizer=self.kernel_regularizer,
        #         kernel_constraint=self.kernel_constraint,
        #         use_bias=self.use_bias,
        #         bias_initializer=self.bias_initializer,
        #         bias_regularizer=self.bias_regularizer,
        #         bias_constraint=self.bias_constraint,
			# 	activation=self.recurrent_activation)
        # c_kernel = Dense(output_dim,
        #         kernel_initializer=self.kernel_initializer,
        #         kernel_regularizer=self.kernel_regularizer,
        #         kernel_constraint=self.kernel_constraint,
        #         use_bias=self.use_bias,
        #         bias_initializer=self.bias_initializer,
        #         bias_regularizer=self.bias_regularizer,
        #         bias_constraint=self.bias_constraint,
			# 	activation=self.recurrent_activation)
        # x_kernel_r = Dense(output_dim,
        #         kernel_initializer=self.kernel_initializer,
        #         kernel_regularizer=self.kernel_regularizer,
        #         kernel_constraint=self.kernel_constraint,
        #         use_bias=self.use_bias,
        #         bias_initializer=self.bias_initializer,
        #         bias_regularizer=self.bias_regularizer,
        #         bias_constraint=self.bias_constraint,
			# 	activation=self.recurrent_activation)
        # h_kernel_r = Dense(output_dim,
        #         kernel_initializer=self.kernel_initializer,
        #         kernel_regularizer=self.kernel_regularizer,
        #         kernel_constraint=self.kernel_constraint,
        #         use_bias=self.use_bias,
        #         bias_initializer=self.bias_initializer,
        #         bias_regularizer=self.bias_regularizer,
        #         bias_constraint=self.bias_constraint,
			# 	activation=self.recurrent_activation)
        # c_kernel_r = Dense(output_dim,
        #         kernel_initializer=self.kernel_initializer,
        #         kernel_regularizer=self.kernel_regularizer,
        #         kernel_constraint=self.kernel_constraint,
        #         use_bias=self.use_bias,
        #         bias_initializer=self.bias_initializer,
        #         bias_regularizer=self.bias_regularizer,
        #         bias_constraint=self.bias_constraint,
			# 	activation=self.recurrent_activation)

        f_transfer_t_kernel = Dense(output_dim,
                kernel_initializer='zeros',
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer='zeros',
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        i_transfer_t_kernel = Dense(output_dim,
                kernel_initializer='zeros',
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer='zeros',
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        # x_transfer_t_kernel = Dense(output_dim,
        #         kernel_initializer='zeros',
        #         kernel_regularizer=self.kernel_regularizer,
        #         kernel_constraint=self.kernel_constraint,
        #         use_bias=self.use_bias,
        #         bias_initializer='zeros',
        #         bias_regularizer=self.bias_regularizer,
        #         bias_constraint=self.bias_constraint)

        f_transfer_t_recurrentkernel = Dense(output_dim,
                kernel_initializer='zeros',
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer='zeros',
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        i_transfer_t_recurrentkernel = Dense(output_dim,
                kernel_initializer='zeros',
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer='zeros',
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        # x_transfer_t_recurrentkernel = Dense(output_dim,
        #         kernel_initializer='zeros',
        #         kernel_regularizer=self.kernel_regularizer,
        #         kernel_constraint=self.kernel_constraint,
        #         use_bias=self.use_bias,
        #         bias_initializer='zeros',
        #         bias_regularizer=self.bias_regularizer,
        #         bias_constraint=self.bias_constraint)

        f_transfer_s_kernel = Dense(output_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        i_transfer_s_kernel = Dense(output_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        # x_transfer_s_kernel = Dense(output_dim,
        #         kernel_initializer=self.kernel_initializer,
        #         kernel_regularizer=self.kernel_regularizer,
        #         kernel_constraint=self.kernel_constraint,
        #         use_bias=self.use_bias,
        #         bias_initializer=self.bias_initializer,
        #         bias_regularizer=self.bias_regularizer,
        #         bias_constraint=self.bias_constraint)

        f_transfer_s_recurrentkernel = Dense(output_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        i_transfer_s_recurrentkernel = Dense(output_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        # x_transfer_s_recurrentkernel = Dense(output_dim,
        #         kernel_initializer=self.kernel_initializer,
        #         kernel_regularizer=self.kernel_regularizer,
        #         kernel_constraint=self.kernel_constraint,
        #         use_bias=self.use_bias,
        #         bias_initializer=self.bias_initializer,
        #         bias_regularizer=self.bias_regularizer,
        #         bias_constraint=self.bias_constraint)
        # x_t=x_rx #Activation('tanh')(add([f_transfer_t_kernel(x_rx),f_transfer_t_recurrentkernel(h_tm1x)]))
        # x_s=x_lx  #Activation('tanh')(add([f_transfer_s_kernel(x_lx),f_transfer_s_recurrentkernel(x_h)]))



        f_t=Activation('hard_sigmoid')(add([f_transfer_t_kernel(x_rx),f_transfer_t_recurrentkernel(h_tm1x)]))
        i_t=Activation('hard_sigmoid')(add([i_transfer_t_kernel(x_rx),i_transfer_t_recurrentkernel(h_tm1x)]))
        c_t=multiply([c_tm1x,f_t])
        h_t=multiply([h_tm1x,i_t])

        f_s=Activation('hard_sigmoid')(add([f_transfer_s_kernel(x_lx),f_transfer_s_recurrentkernel(x_h)]))
        i_s=Activation('hard_sigmoid')(add([i_transfer_s_kernel(x_lx),i_transfer_s_recurrentkernel(x_h)]))
        c_s=multiply([x_c,f_s])
        h_s=multiply([x_h,i_s])

        c_tm1=add([c_s,c_t])
        h_tm1=add([h_s,h_t])
        x=add([x_lx,x_rx])

        '''x_l_gated=Activation(self.activation)(multiply([x_lx,x_kernel(concatenate([x_lx,x_h]))]))
        h_1_gated=Activation(self.recurrent_activation)(multiply([x_h ,h_kernel(concatenate([x_lx,x_h]))]))
        c_1_gated=Activation(self.recurrent_activation)(multiply([x_c,c_kernel(concatenate([x_lx,x_h]))]))
        x_r_gated=Activation(self.activation)(multiply([x_rx,x_kernel_r(concatenate([x_rx,h_tm1x]))]))
        h_r_gated=Activation(self.recurrent_activation)(multiply([h_tm1x,h_kernel_r(concatenate([x_rx,h_tm1x]))]))
        c_r_gated=Activation(self.recurrent_activation)(multiply([c_tm1x,c_kernel_r(concatenate([x_rx,h_tm1x]))]))
        x=add([x_r_gated,x_l_gated])
        h_tm1=add([h_r_gated,h_1_gated])
        c_tm1=add([c_r_gated,c_1_gated])'''
        '''x=x_kernel(concatenate([x_lx,x_rx]))
        h_tm1=h_kernel(concatenate([h_tm1x,x_h]))
        c_tm1=c_kernel(concatenate([c_tm1x,x_c]))
        #h_tm1=h_tm1x
        #c_tm1=c_tm1x'''
        kernel = Dense(output_dim * 4,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(output_dim * 4,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=True)

        kernel_out = kernel(x)
        recurrent_kernel_out = recurrent_kernel(h_tm1)
        kernel_i, kernel_f, kernel_c, kernel_o = get_slices(kernel_out, 4)
        recurrent_kernel_i, recurrent_kernel_f, recurrent_kernel_c, recurrent_kernel_o = get_slices(recurrent_kernel_out, 4)

        # inputs_i = x
        # inputs_f = x
        # inputs_c = x
        # inputs_o = x
        #

        x_i = kernel_i
        x_f = kernel_f
        x_c = kernel_c
        x_o = kernel_o

        # if False:#self.use_bias:
        #     self.bias_i = self.bias[:45]
        #     self.bias_f = self.bias[45: 45 * 2]
        #     self.bias_c = self.bias[45 * 2: 45 * 3]
        #     self.bias_o = self.bias[45 * 3:]
        #
        #     x_i = add([x_i, self.bias_i])
        #     x_f = add([x_f, self.bias_f])
        #     x_c = add([x_c, self.bias_c])
        #     x_o = add([x_o, self.bias_o])

        # if False:#0 < self.recurrent_dropout < 1.:
        #     h_tm1_i = h_tm1 * rec_dp_mask[0]
        #     h_tm1_f = h_tm1 * rec_dp_mask[1]
        #     h_tm1_c = h_tm1 * rec_dp_mask[2]
        #     h_tm1_o = h_tm1 * rec_dp_mask[3]
        # else:
        #     h_tm1_i = h_tm1
        #     h_tm1_f = h_tm1
        #     h_tm1_c = h_tm1
        #     h_tm1_o = h_tm1

        i = Activation(self.recurrent_activation)(add([x_i ,recurrent_kernel_i]))
        f = Activation(self.recurrent_activation)(add([x_f ,recurrent_kernel_f]))
        temp= Activation(self.activation)(add([x_c, recurrent_kernel_c]))
        c = add([multiply([f, c_tm1]) , multiply([ i ,temp ]) ])
        o = Activation(self.recurrent_activation)(add([x_o , recurrent_kernel_o]))
        temp2=Activation(self.activation)(c)
        h = multiply([o , temp2])
        temp2=Activation(self.activation)(c)
        h = multiply([o , temp2])

      
        return Model([xx, h_tm1x, c_tm1x], [h, Identity()(h), c])


class KerasLSTMCell_right_concatenate(ExtendedRNNCell):

    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)

        xx = Input(batch_shape=input_shape)
        h_tm1x = Input(batch_shape=output_shape)
        c_tm1x = Input(batch_shape=output_shape)
        x_lx=Lambda(lambda x: x[:,:45])(xx)
        x_h=Lambda(lambda x: x[:,45:90])(xx)
        x_c=Lambda(lambda x: x[:,90:135])(xx)
        x_rx=Lambda(lambda x: x[:,135:])(xx)

        c_transfer_kernel = Dense(output_dim,
                kernel_initializer= 'zeros',
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer='zeros',
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        h_transfer_kernel = Dense(output_dim,
                kernel_initializer='zeros',
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer='zeros',
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        x_transfer_kernel = Dense(output_dim,
                kernel_initializer='zeros',
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer='zeros',
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)

        c_kernel = Dense(output_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        h_kernel = Dense(output_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)
        x_kernel = Dense(output_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint)

        c_l=c_transfer_kernel(x_c)
        h_l=h_transfer_kernel(x_h)
        x_l=x_transfer_kernel(x_lx)

        c_r=c_kernel(c_tm1x)
        h_r=h_kernel(h_tm1x)
        x_r=x_kernel(x_rx)

        c_tm1=add([c_l,c_r])
        h_tm1=add([h_l,h_r])
        x=add([x_l,x_r])

        '''c_tm1=add([c_l,c_tm1x])
        h_tm1=add([x_h,h_tm1x])
        x=concatenate([x_lx,x_rx])'''

        '''x_l_gated=Activation(self.activation)(multiply([x_lx,x_kernel(concatenate([x_lx,x_h]))]))
        h_1_gated=Activation(self.recurrent_activation)(multiply([x_h ,h_kernel(concatenate([x_lx,x_h]))]))
        c_1_gated=Activation(self.recurrent_activation)(multiply([x_c,c_kernel(concatenate([x_lx,x_h]))]))
        x_r_gated=Activation(self.activation)(multiply([x_rx,x_kernel_r(concatenate([x_rx,h_tm1x]))]))
        h_r_gated=Activation(self.recurrent_activation)(multiply([h_tm1x,h_kernel_r(concatenate([x_rx,h_tm1x]))]))
        c_r_gated=Activation(self.recurrent_activation)(multiply([c_tm1x,c_kernel_r(concatenate([x_rx,h_tm1x]))]))
        x=add([x_r_gated,x_l_gated])
        h_tm1=add([h_r_gated,h_1_gated])
        c_tm1=add([c_r_gated,c_1_gated])'''

        #add dropout

        '''x=x_kernel(concatenate([x_lx,x_rx]))
        h_tm1=h_kernel(concatenate([h_tm1x,x_h]))
        c_tm1=c_kernel(concatenate([c_tm1x,x_c]))
        #h_tm1=h_tm1x
        #c_tm1=c_tm1x'''

        kernel = Dense(output_dim * 4,
                       kernel_initializer=self.kernel_initializer,
                       kernel_regularizer=self.kernel_regularizer,
                       kernel_constraint=self.kernel_constraint,
                       use_bias=self.use_bias,
                       bias_initializer=self.bias_initializer,
                       bias_regularizer=self.bias_regularizer,
                       bias_constraint=self.bias_constraint)
        recurrent_kernel = Dense(output_dim * 4,
                                 kernel_initializer=self.recurrent_initializer,
                                 kernel_regularizer=self.recurrent_regularizer,
                                 kernel_constraint=self.recurrent_constraint,
                                 use_bias=True)

        kernel_out = kernel(x)
        recurrent_kernel_out = recurrent_kernel(h_tm1)
        kernel_i, kernel_f, kernel_c, kernel_o = get_slices(kernel_out, 4)
        recurrent_kernel_i, recurrent_kernel_f, recurrent_kernel_c, recurrent_kernel_o = get_slices(recurrent_kernel_out, 4)

        # inputs_i = x
        # inputs_f = x
        # inputs_c = x
        # inputs_o = x
        #

        x_i = kernel_i
        x_f = kernel_f
        x_c = kernel_c
        x_o = kernel_o

        if False:#self.use_bias:
            self.bias_i = self.bias[:45]
            self.bias_f = self.bias[45: 45 * 2]
            self.bias_c = self.bias[45 * 2: 45 * 3]
            self.bias_o = self.bias[45 * 3:]

            x_i = add([x_i, self.bias_i])
            x_f = add([x_f, self.bias_f])
            x_c = add([x_c, self.bias_c])
            x_o = add([x_o, self.bias_o])

        if False:#0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        i = Activation(self.recurrent_activation)(add([x_i ,recurrent_kernel_i]))
        f = Activation(self.recurrent_activation)(add([x_f ,recurrent_kernel_f]))
        temp= Activation(self.activation)(add([x_c, recurrent_kernel_c]))
        c = add([multiply([f, c_tm1]) , multiply([ i ,temp ]) ])
        o = Activation(self.recurrent_activation)(add([x_o , recurrent_kernel_o]))
        temp2=Activation(self.activation)(c)
        h = multiply([o , temp2])
        temp2=Activation(self.activation)(c)
        h = multiply([o , temp2])

      
        return Model([xx, h_tm1x, c_tm1x], [h, Identity()(h), c])