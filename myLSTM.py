# -*- coding: utf-8 -*-
# @Time    : 18-3-20 上午10:57
# @Author  : Icy Shen
# @Email   : SAH1949@126.com
from keras.layers import Input, Lambda, add, concatenate, Activation, multiply, constraints, activations, regularizers, Dense,Layer
from keras.models import Model
import keras.backend as K
from recurrentshop.engine import RNNCell
from keras import initializers
import numpy as np
def _slice(x, dim, index):
    return x[:, index * dim: dim * (index + 1)]


def get_slices(x, n):
    dim = int(K.int_shape(x)[1] / n)
    return [Lambda(_slice, arguments={'dim': dim, 'index': i}, output_shape=lambda s: (s[0], dim))(x) for i in range(n)]

def _generate_dropout_ones(inputs, dims):
    # Currently, CTNK can't instantiate `ones` with symbolic shapes.
    # Will update workaround once CTNK supports it.
    if K.backend() == 'cntk':
        ones = K.ones_like(K.reshape(inputs[:, 0], (-1, 1)))
        return K.tile(ones, (1, dims))
    else:
        return K.ones((K.shape(inputs)[0], dims))

def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)


class Identity(Layer):
    def call(self, x):
        return x + 0.

class ExtendedRNNCell(RNNCell):

    def __init__(self, units=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 unit_forget_bias=True,
                 dropout=0.,
                 recurrent_dropout = 0.,
                 implementation = 1,
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
        self.unit_forget_bias = unit_forget_bias
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.dropout = min(1., max(0., dropout))
        self.implementation = implementation
        self._trainable_weights = []
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
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


# class KerasLSTMCell_right_concatenate(ExtendedRNNCell):
#
#     def build_model(self, input_shape, training=None):
#         input_dim = input_shape[-1]
#         inputs = Input(batch_shape=input_shape)
#         output_shape = (input_shape[0], self.output_dim)
#         h_tm1x = Input(batch_shape=output_shape)
#         c_tm1x = Input(batch_shape=output_shape)
#
#         kernel = self.add_weight(shape=(input_dim, self.output_dim * 4),
#                                       name='kernel',
#                                       initializer=self.kernel_initializer,
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#         recurrent_kernel = self.add_weight(
#             shape=(self.output_dim, self.output_dim * 4),
#             name='recurrent_kernel',
#             initializer=self.recurrent_initializer,
#             regularizer=self.recurrent_regularizer,
#             constraint=self.recurrent_constraint)
#
#         if self.use_bias:
#             if self.unit_forget_bias:
#                 def bias_initializer(_, *args, **kwargs):
#                     return K.concatenate([
#                         self.bias_initializer((self.output_dim,), *args, **kwargs),
#                         initializers.Ones()((self.output_dim,), *args, **kwargs),
#                         self.bias_initializer((self.output_dim * 2,), *args, **kwargs),
#                     ])
#             else:
#                 bias_initializer = self.bias_initializer
#             self.bias = self.add_weight(shape=(self.output_dim * 4,),
#                                         name='bias',
#                                         initializer=bias_initializer,
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#
#         kernel_i = kernel[:, :self.output_dim] #input
#         kernel_f = kernel[:, self.output_dim: self.output_dim * 2] #
#         kernel_c = kernel[:, self.output_dim * 2: self.output_dim * 3]
#         kernel_o = kernel[:, self.output_dim * 3:]
#
#         recurrent_kernel_i = recurrent_kernel[:, :self.output_dim]#h
#         recurrent_kernel_f = recurrent_kernel[:, self.output_dim: self.output_dim * 2]
#         recurrent_kernel_c = recurrent_kernel[:, self.output_dim * 2: self.output_dim * 3]
#         recurrent_kernel_o = recurrent_kernel[:, self.output_dim * 3:]
#
#         if self.use_bias:
#             self.bias_i = self.bias[:self.output_dim]
#             self.bias_f = self.bias[self.output_dim: self.output_dim * 2]
#             self.bias_c = self.bias[self.output_dim * 2: self.output_dim * 3]
#             self.bias_o = self.bias[self.output_dim * 3:]
#         else:
#             self.bias_i = None
#             self.bias_f = None
#             self.bias_c = None
#             self.bias_o = None
#         # self.built = True
#         if 0 < self.dropout < 1 and self._dropout_mask is None:
#             self._dropout_mask = _generate_dropout_mask(
#                 _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
#                 self.dropout,
#                 training=training,
#                 count=4)
#         if (0 < self.recurrent_dropout < 1 and
#                 self._recurrent_dropout_mask is None):
#             self._recurrent_dropout_mask = _generate_dropout_mask(
#                 _generate_dropout_ones(inputs, self.units),
#                 self.recurrent_dropout,
#                 training=training,
#                 count=4)
#
#         # dropout matrices for input units
#         dp_mask = self._dropout_mask
#         # dropout matrices for recurrent units
#         rec_dp_mask = self._recurrent_dropout_mask
#
#         # h_tm1 = states[0]  # previous memory state
#         # c_tm1 = states[1]  # previous carry state
#
#         c_tm1 = c_tm1x
#         h_tm1 = h_tm1x
#         # x = concatenate([x_lx, x_rx])
#
#         if self.implementation == 1:
#             if 0 < self.dropout < 1.:
#                 inputs_i = inputs * dp_mask[0]
#                 inputs_f = inputs * dp_mask[1]
#                 inputs_c = inputs * dp_mask[2]
#                 inputs_o = inputs * dp_mask[3]
#             else:
#                 inputs_i = inputs
#                 inputs_f = inputs
#                 inputs_c = inputs
#                 inputs_o = inputs
#             x_i = K.dot(inputs_i, kernel_i)
#             x_f = K.dot(inputs_f, kernel_f)
#             x_c = K.dot(inputs_c, kernel_c)
#             x_o = K.dot(inputs_o, kernel_o)
#             if self.use_bias:
#                 x_i = K.bias_add(x_i, self.bias_i)
#                 x_f = K.bias_add(x_f, self.bias_f)
#                 x_c = K.bias_add(x_c, self.bias_c)
#                 x_o = K.bias_add(x_o, self.bias_o)
#
#             if 0 < self.recurrent_dropout < 1.:
#                 h_tm1_i = h_tm1 * rec_dp_mask[0]
#                 h_tm1_f = h_tm1 * rec_dp_mask[1]
#                 h_tm1_c = h_tm1 * rec_dp_mask[2]
#                 h_tm1_o = h_tm1 * rec_dp_mask[3]
#             else:
#                 h_tm1_i = h_tm1
#                 h_tm1_f = h_tm1
#                 h_tm1_c = h_tm1
#                 h_tm1_o = h_tm1
#             i = self.recurrent_activation(x_i + K.dot(h_tm1_i,
#                                                       recurrent_kernel_i))
#             f = self.recurrent_activation(x_f + K.dot(h_tm1_f,
#                                                       recurrent_kernel_f))
#             c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
#                                                             recurrent_kernel_c))
#             o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
#                                                       recurrent_kernel_o))
#         else:
#             if 0. < self.dropout < 1.:
#                 inputs *= dp_mask[0]
#             z = K.dot(inputs, kernel)
#             if 0. < self.recurrent_dropout < 1.:
#                 h_tm1 *= rec_dp_mask[0]
#             z += K.dot(h_tm1, recurrent_kernel)
#             if self.use_bias:
#                 z = K.bias_add(z, self.bias)
#
#             z0 = z[:, :self.output_dim]
#             z1 = z[:, self.output_dim: 2 * self.output_dim]
#             z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
#             z3 = z[:, 3 * self.output_dim:]
#
#             i = self.recurrent_activation(z0)
#             f = self.recurrent_activation(z1)
#             c = f * c_tm1 + i * self.activation(z2)
#             o = self.recurrent_activation(z3)
#
#         h = o * self.activation(c)
#         if 0 < self.dropout + self.recurrent_dropout:
#             if training is None:
#                 h._uses_learning_phase = True
#         Identity = Lambda(Identity())
#         return Model([inputs, h_tm1x, c_tm1x], [h, Identity()(h), c])


# class KerasLSTMCell_right_concatenate(ExtendedRNNCell):
#
#     def build_model(self, input_shape, training = None):
#         output_dim = self.output_dim
#         input_dim = input_shape[-1]
#         output_shape = (input_shape[0], output_dim)
#         x = Input(batch_shape=input_shape)
#
#         x_r = Lambda(lambda x: x[:, (self.output_dim * 3):])(x)
#         h_tm1x = Input(batch_shape=output_shape)
#         c_tm1x = Input(batch_shape=output_shape)
#
#         x_l = Lambda(lambda x: x[:,:self.output_dim])(x)
#         h_l = Lambda(lambda x: x[:, self.output_dim:(self.output_dim * 2)])(x)
#         c_l = Lambda(lambda x: x[:, (self.output_dim * 2):(self.output_dim * 3)])(x)
#         episilon = np.random.random()
#
#         if (episilon > 0):
#             # kernel_xr = Dense(output_dim,
#             #                kernel_initializer=self.kernel_initializer,
#             #                kernel_regularizer=self.kernel_regularizer,
#             #                kernel_constraint=self.kernel_constraint,
#             #                use_bias=self.use_bias,
#             #                bias_initializer=self.bias_initializer,
#             #                bias_regularizer=self.bias_regularizer,
#             #                bias_constraint=self.bias_constraint)
#             # recurrent_kernel_hr = Dense(output_dim,
#             #                          kernel_initializer=self.recurrent_initializer,
#             #                          kernel_regularizer=self.recurrent_regularizer,
#             #                          kernel_constraint=self.recurrent_constraint,
#             #                          use_bias=False)
#             # recurrent_kernel_cr = Dense(output_dim,
#             #                             kernel_initializer=self.recurrent_initializer,
#             #                             kernel_regularizer=self.recurrent_regularizer,
#             #                             kernel_constraint=self.recurrent_constraint,
#             #                             use_bias=False)
#             # xr = kernel_xr(x_r)
#             # hr = recurrent_kernel_hr(h_tm1)
#             # cr = recurrent_kernel_cr(c_tm1)
#             #
#             # kernel_l = Dense(output_dim * 4,
#             #                kernel_initializer=self.kernel_initializer,
#             #                kernel_regularizer=self.kernel_regularizer,
#             #                kernel_constraint=self.kernel_constraint,
#             #                use_bias=self.use_bias,
#             #                bias_initializer=self.bias_initializer,
#             #                bias_regularizer=self.bias_regularizer,
#             #                bias_constraint=self.bias_constraint)
#             # recurrent_kernel_l = Dense(output_dim * 4,
#             #                          kernel_initializer=self.recurrent_initializer,
#             #                          kernel_regularizer=self.recurrent_regularizer,
#             #                          kernel_constraint=self.recurrent_constraint,
#             #                          use_bias=False)
#             #
#             # kernel_out_l = kernel_l(x_l)
#             # recurrent_kernel_out_l = recurrent_kernel_l(h_l)
#             #
#             # x0_l, x1_l, x2_l, x3_l = get_slices(kernel_out_l, 4)
#             # r0_l, r1_l, r2_l, r3_l = get_slices(recurrent_kernel_out_l, 4)
#
#             # f_l = add([x0_l, r0_l])
#             # f_l = Activation(self.recurrent_activation)(f_l)
#             #
#             # i_l = add([x1_l, r1_l])
#             # i_l = Activation(self.recurrent_activation)(i_l)
#             #
#             # c_prime_l = add([x2_l, r2_l])
#             # c_prime_l = Activation(self.activation)(c_prime_l)
#             # c_l = add([multiply([f_l, c_l]), multiply([i_l, c_prime_l])])
#             # c_l_o = Activation(self.activation)(c_l)
#             #
#             # o_l = add([x3_l, r3_l])
#             # h_l_o = multiply([o_l, c_l])
#
#             # x_all = add([xr, x_l])
#             # h_tm1 = add([hr, h_l_o])
#             # c_tm1 = add([cr, c_l_o])
#             c_transfer_kernel = Dense(output_dim,
#                                       kernel_initializer='zeros',
#                                       kernel_regularizer=self.kernel_regularizer,
#                                       kernel_constraint=self.kernel_constraint,
#                                       use_bias=self.use_bias,
#                                       bias_initializer='zeros',
#                                       bias_regularizer=self.bias_regularizer,
#                                       bias_constraint=self.bias_constraint)
#             h_transfer_kernel = Dense(output_dim,
#                                       kernel_initializer='zeros',
#                                       kernel_regularizer=self.kernel_regularizer,
#                                       kernel_constraint=self.kernel_constraint,
#                                       use_bias=self.use_bias,
#                                       bias_initializer='zeros',
#                                       bias_regularizer=self.bias_regularizer,
#                                       bias_constraint=self.bias_constraint)
#             x_transfer_kernel = Dense(output_dim,
#                                       kernel_initializer='zeros',
#                                       kernel_regularizer=self.kernel_regularizer,
#                                       kernel_constraint=self.kernel_constraint,
#                                       use_bias=self.use_bias,
#                                       bias_initializer='zeros',
#                                       bias_regularizer=self.bias_regularizer,
#                                       bias_constraint=self.bias_constraint)
#
#             c_kernel = Dense(output_dim,
#                              kernel_initializer=self.kernel_initializer,
#                              kernel_regularizer=self.kernel_regularizer,
#                              kernel_constraint=self.kernel_constraint,
#                              use_bias=self.use_bias,
#                              bias_initializer=self.bias_initializer,
#                              bias_regularizer=self.bias_regularizer,
#                              bias_constraint=self.bias_constraint)
#             h_kernel = Dense(output_dim,
#                              kernel_initializer=self.kernel_initializer,
#                              kernel_regularizer=self.kernel_regularizer,
#                              kernel_constraint=self.kernel_constraint,
#                              use_bias=self.use_bias,
#                              bias_initializer=self.bias_initializer,
#                              bias_regularizer=self.bias_regularizer,
#                              bias_constraint=self.bias_constraint)
#             x_kernel = Dense(output_dim,
#                              kernel_initializer=self.kernel_initializer,
#                              kernel_regularizer=self.kernel_regularizer,
#                              kernel_constraint=self.kernel_constraint,
#                              use_bias=self.use_bias,
#                              bias_initializer=self.bias_initializer,
#                              bias_regularizer=self.bias_regularizer,
#                              bias_constraint=self.bias_constraint)
#
#             c_l = c_transfer_kernel(c_l)
#             h_l = h_transfer_kernel(h_l)
#             x_l = x_transfer_kernel(x_l)
#
#             c_r = c_kernel(c_tm1x)
#             h_r = h_kernel(h_tm1x)
#             x_r = x_kernel(x_r)
#
#             c_tm1 = add([c_l, c_r])
#             h_tm1 = add([h_l, h_r])
#             x_all = add([x_l, x_r])
#
#             # x_all = add([xr, xl])
#             # h_tm1 = add([hr, hl])
#             # c_tm1 = add([cr, cl])
#
#         else:
#             x_all = x_r
#         kernel = Dense(output_dim * 4,
#                        kernel_initializer=self.kernel_initializer,
#                        kernel_regularizer=self.kernel_regularizer,
#                        kernel_constraint=self.kernel_constraint,
#                        use_bias=self.use_bias,
#                        bias_initializer=self.bias_initializer,
#                        bias_regularizer=self.bias_regularizer,
#                        bias_constraint=self.bias_constraint)
#         recurrent_kernel = Dense(output_dim * 4,
#                                  kernel_initializer=self.recurrent_initializer,
#                                  kernel_regularizer=self.recurrent_regularizer,
#                                  kernel_constraint=self.recurrent_constraint,
#                                  use_bias=False)
#         kernel_out = kernel(x_all)
#         recurrent_kernel_out = recurrent_kernel(h_tm1)
#
#         x0, x1, x2, x3 = get_slices(kernel_out, 4)
#         r0, r1, r2, r3 = get_slices(recurrent_kernel_out, 4)
#
#         f = add([x0, r0])
#         f = Activation(self.recurrent_activation)(f)
#
#         i = add([x1, r1])
#         i = Activation(self.recurrent_activation)(i)
#
#         c_prime = add([x2, r2])
#         c_prime = Activation(self.activation)(c_prime)
#         c = add([multiply([f, c_tm1]), multiply([i, c_prime])])
#         c = Activation(self.activation)(c)
#         o = add([x3, r3])
#         h = multiply([o, c])
#         y = concatenate([h, h_tm1, c_tm1])
#         return Model([x, h_tm1, c_tm1], [y, Identity()(h), c])
import numpy as np
class KerasLSTMCell_right_concatenate(ExtendedRNNCell):
    def build_model(self, input_shape):
        output_dim = self.output_dim
        # input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)

        lrx = Input(batch_shape=input_shape)
        h_r = Input(batch_shape=output_shape)
        c_r = Input(batch_shape=output_shape)
        episilon = np.random.random()
        bound = 0
        if episilon > bound:
            l_x = Lambda(lambda x: x[:, :self.output_dim])(lrx )
            h_l = Lambda(lambda x: x[:, self.output_dim:(self.output_dim * 2)])(lrx )
            c_l = Lambda(lambda x: x[:, (self.output_dim * 2):(self.output_dim * 3)])(lrx )
            r_x = Lambda(lambda x: x[:, (self.output_dim * 3):])(lrx )
            # c_transfer_kernel = Dense(output_dim,
            #                           kernel_initializer='zeros',
            #                           kernel_regularizer=self.kernel_regularizer,
            #                           kernel_constraint=self.kernel_constraint,
            #                           use_bias=self.use_bias,
            #                           bias_initializer='zeros',
            #                           bias_regularizer=self.bias_regularizer,
            #                           bias_constraint=self.bias_constraint)
            # h_transfer_kernel = Dense(output_dim,
            #                           kernel_initializer='zeros',
            #                           kernel_regularizer=self.kernel_regularizer,
            #                           kernel_constraint=self.kernel_constraint,
            #                           use_bias=self.use_bias,
            #                           bias_initializer='zeros',
            #                           bias_regularizer=self.bias_regularizer,
            #                           bias_constraint=self.bias_constraint)
            # x_transfer_kernel = Dense(output_dim,
            #                           kernel_initializer='zeros',
            #                           kernel_regularizer=self.kernel_regularizer,
            #                           kernel_constraint=self.kernel_constraint,
            #                           use_bias=self.use_bias,
            #                           bias_initializer='zeros',
            #                           bias_regularizer=self.bias_regularizer,
            #                           bias_constraint=self.bias_constraint)
            #
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
            # x_kernel = Dense(output_dim,
            #                  kernel_initializer=self.kernel_initializer,
            #                  kernel_regularizer=self.kernel_regularizer,
            #                  kernel_constraint=self.kernel_constraint,
            #                  use_bias=self.use_bias,
            #                  bias_initializer=self.bias_initializer,
            #                  bias_regularizer=self.bias_regularizer,
            #                  bias_constraint=self.bias_constraint)
            # cl = c_transfer_kernel(c_l)
            # hl = h_transfer_kernel(h_l)
            # lx = x_transfer_kernel(l_x)
            # rx = x_kernel(r_x)
            cr = c_kernel(c_r)
            hr = h_kernel(h_r)
            kernel_l = Dense(output_dim * 4,
                           kernel_initializer=self.kernel_initializer,
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_constraint=self.kernel_constraint,
                           use_bias=self.use_bias,
                           bias_initializer=self.bias_initializer,
                           bias_regularizer=self.bias_regularizer,
                           bias_constraint=self.bias_constraint)
            recurrent_kernel_l = Dense(output_dim * 4,
                                     kernel_initializer=self.recurrent_initializer,
                                     kernel_regularizer=self.recurrent_regularizer,
                                     kernel_constraint=self.recurrent_constraint,
                                     use_bias=True)

            kernel_out_l = kernel_l(l_x)
            recurrent_kernel_out_l = recurrent_kernel_l(h_l)
            l_i, l_f, l_c, l_o = get_slices(kernel_out_l, 4)
            recurrent_l_i, recurrent_l_f, recurrent_l_c, recurrent_l_o = get_slices(
                recurrent_kernel_out_l, 4)


            f_l = add([l_f, recurrent_l_f])
            f_l = Activation(self.recurrent_activation)(f_l)

            i_l = add([l_i, recurrent_l_i])
            i_l = Activation(self.recurrent_activation)(i_l)

            c_prime_l = add([l_c, recurrent_l_c])
            c_prime_l = Activation(self.activation)(c_prime_l)
            c_l = add([multiply([f_l, c_l]), multiply([i_l, c_prime_l])])
            c_l_o = Activation(self.activation)(c_l)

            o_l = add([l_o, recurrent_l_o])
            h_l_o = multiply([o_l, c_l])

            c_tm1 = add([c_l_o, cr])
            h_tm1 = add([h_l_o, hr])
            x = r_x
            # x = add([rx, lx])
            # c_tm1 = add([cl, cr])
            # h_tm1 = add([hl, hr])
            # x = add([rx, lx])
        else:
            c_tm1 = c_r
            h_tm1 = h_r
            x = lrx

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
        recurrent_kernel_i, recurrent_kernel_f, recurrent_kernel_c, recurrent_kernel_o = get_slices(
            recurrent_kernel_out, 4)
        i = add([kernel_i, recurrent_kernel_i])
        i = Activation(self.recurrent_activation)(i)

        f = add([kernel_f, recurrent_kernel_f])
        f = Activation(self.recurrent_activation)(f)

        c_prime = add([kernel_c, recurrent_kernel_c])
        c_prime = Activation(self.activation)(c_prime)
        c = add([multiply([f, c_tm1]), multiply([i, c_prime])])

        o = add([kernel_o, recurrent_kernel_o])
        o = Activation(self.recurrent_activation)(o)
        c = Activation(self.activation)(c)
        h = multiply([o, c])
        return Model([lrx , h_r, c_r], [h, Identity()(h), c])

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
        recurrent_kernel_i, recurrent_kernel_f, recurrent_kernel_c, recurrent_kernel_o = get_slices(
            recurrent_kernel_out, 4)

        inputs_i = x
        inputs_f = x
        inputs_c = x
        inputs_o = x

        x_i = kernel_i
        x_f = kernel_f
        x_c = kernel_c
        x_o = kernel_o


        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1

        i = Activation(self.recurrent_activation)(add([x_i, recurrent_kernel_i]))
        f = Activation(self.recurrent_activation)(add([x_f, recurrent_kernel_f]))
        temp = Activation(self.activation)(add([x_c, recurrent_kernel_c]))
        c = add([multiply([f, c_tm1]), multiply([i, temp])])
        o = Activation(self.recurrent_activation)(add([x_o, recurrent_kernel_o]))
        c = Activation(self.activation)(c)
        h = multiply([o, c])

        y = concatenate([h, h_tm1, c_tm1])

        return Model([x, h_tm1, c_tm1], [y, Identity()(h), c])