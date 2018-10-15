 ##TODO
import dynet_config
dynet_config.set(mem='3072', random_seed=1314159)
import dynet as dy
import math
import numpy as np
import time
from utils import bio2ot


class WDEmb:
    def __init__(self, pc, n_words, dim_w, pretrained_embeddings=None):
        self.pc = pc.add_subcollection()
        self.n_words = n_words
        self.dim_w = dim_w
        self.W = self.pc.add_lookup_parameters((self.n_words, self.dim_w))
        if pretrained_embeddings is not None:
            print("Use pretrained word embeddings")
            self.W.init_from_array(pretrained_embeddings)

    def parametrize(self):
        """
        note: lookup parameters do not need parametrization
        :return:
        """
        pass

    def __call__(self, xs):
        """

        :param xs: a list of ngrams (or words if win is set to 1)
        :return: embeddings looked from tables
        """
        embeddings = [dy.concatenate([self.W[w] for w in ngram]) for ngram in xs]
        return embeddings


class LSTM:
    def __init__(self, pc, n_in, n_out, dropout_rate):
        """
        LSTM constructor
        :param pc: parameter collection
        :param n_in:
        :param n_out:
        :param dropout_rate: dropout rate
        """
        self.n_in = n_in
        self.n_out = n_out
        self.dropout_rate = dropout_rate
        self.pc = pc.add_subcollection()

        self._W = self.pc.add_parameters((4 * self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        self._U = self.pc.add_parameters((4 * self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._b = self.pc.add_parameters((4 * self.n_out), init=dy.ConstInitializer(0.0))

    def parametrize(self):
        """

        :return:
        """
        self.W = dy.parameter(self._W)
        self.U = dy.parameter(self._U)
        self.b = dy.parameter(self._b)

    def __call__(self, inputs, is_train=True):
        """

        :param inputs: input word embeddings
        :param is_train: train flag, used for dropout
        :return:
        """
        seq_len = len(inputs)
        h = dy.zeros((self.n_out,))
        c = dy.zeros((self.n_out,))
        H = []
        for t in range(seq_len):
            xt = inputs[t]
            h, c = self.recurrence(xt, h, c, train_flag=is_train)
            H.append(h)
        return H

    def recurrence(self, xt, htm1, ctm1, train_flag):
        """
        RNN recurrence function
        :param xt: x_t
        :param htm1: h_{t-1}
        :param ctm1: c_{t-1}
        :param train_flag:
        :return:
        """
        if train_flag:
            # in the training phase, perform dropout
            W_dropout = dy.dropout(self.W, self.dropout_rate)
        else:
            W_dropout = self.W
        Wx = W_dropout * xt
        Uh = self.U * htm1
        sum_item = Wx + Uh + self.b
        # gate order: ifco
        it = dy.logistic(sum_item[:self.n_out])
        ft = dy.logistic(sum_item[self.n_out:2*self.n_out])
        ct_hat = dy.tanh(sum_item[2*self.n_out:3*self.n_out])
        ot = dy.logistic(sum_item[3*self.n_out:])
        ct = dy.cmult(ctm1, ft) + dy.cmult(ct_hat, it)
        ht = dy.cmult(dy.tanh(ct), ot)
        return ht, ct


class THA:
    # truncated attention to obtain the aspect history
    def __init__(self, pc, n_steps, n_in):
        """

        :param n_steps: number of steps in truncated self-attention
        :param n_in:
        """
        self.pc = pc.add_subcollection()
        self.n_steps = n_steps
        self.n_in = n_in
        self._v = self.pc.add_parameters((self.n_in,), init=dy.UniformInitializer(0.2))
        self._W1 = self.pc.add_parameters((self.n_in, self.n_in), init=dy.UniformInitializer(0.2))
        self._W2 = self.pc.add_parameters((self.n_in, self.n_in), init=dy.UniformInitializer(0.2))
        self._W3 = self.pc.add_parameters((self.n_in, self.n_in), init=dy.UniformInitializer(0.2))

    def parametrize(self):
        """

        :return:
        """
        self.v = dy.parameter(self._v)
        self.W1 = dy.parameter(self._W1)
        self.W2 = dy.parameter(self._W2)
        self.W3 = dy.parameter(self._W3)

    def __call__(self, H):
        seq_len = len(H)
        H_prev_tilde = dy.zeros((self.n_steps, self.n_in))
        H_prev = dy.zeros((self.n_steps, self.n_in))
        H_tilde = []
        aspect_attentions = []
        for t in range(seq_len):
            ht = H[t]
            w_t = []
            for i in range(self.n_steps):
                hi = H_prev[i]
                hi_tilde = H_prev_tilde[i]
                w_t.append(dy.dot_product(self.v, dy.tanh(self.W1*hi + self.W2*ht + self.W3*hi_tilde)))
            # shape: (n_steps,)
            w_t = dy.softmax(dy.concatenate(w_t))
            aspect_attentions.append((t, w_t.npvalue()))
            ht_hat = dy.reshape(dy.transpose(w_t) * H_prev_tilde, d=(self.n_in,))
            ht_tilde = ht + dy.rectify(ht_hat)
            H_prev = dy.concatenate([H_prev[1:], dy.reshape(ht, (1, self.n_in))])
            H_prev_tilde = dy.concatenate([H_prev_tilde[1:], dy.reshape(ht_tilde, (1, self.n_in))])
            H_tilde.append(ht_tilde)
        return H_tilde, aspect_attentions


class ST_bilinear:
    # selective transformation with bi-linear attention
    def __init__(self, pc, dim_asp, dim_opi):
        self.pc = pc.add_subcollection()
        self.dim_asp = dim_asp
        self.dim_opi = dim_opi
        self._W_A = self.pc.add_parameters((2*self.dim_opi, 2*self.dim_asp), init=dy.UniformInitializer(0.2))
        self._W_O = self.pc.add_parameters((2*self.dim_opi, 2*self.dim_opi), init=dy.UniformInitializer(0.2))
        self._b = self.pc.add_parameters((2*self.dim_opi,), init=dy.ConstInitializer(0.0))
        self._W_bilinear = self.pc.add_parameters((2*self.dim_asp, 2*self.dim_opi), init=dy.UniformInitializer(0.2))
        self._b_bilinear = self.pc.add_parameters((1,), init=dy.ConstInitializer(0.0))

    def parametrize(self):
        self.W_A = dy.parameter(self._W_A)
        self.W_O = dy.parameter(self._W_O)
        self.b = dy.parameter(self._b)
        self.W_bilinear = dy.parameter(self._W_bilinear)
        self.b_bilinear = dy.parameter(self._b_bilinear)

    def __call__(self, htA, HO, transform_flag=True):
        """

        :param htA:
        :param HO:
        :param transform_flag: determine if the model needs selective transformation,
        :return:
        """
        seq_len = len(HO)
        HO_hat = []
        Weights = []
        for i in range(seq_len):
            hiO = HO[i]
            if transform_flag:
                hiO_hat = hiO + dy.rectify(self.W_A * htA + self.W_O * hiO + self.b)
            else:
                hiO_hat = hiO
            wi = dy.tanh(dy.transpose(htA) * self.W_bilinear * hiO_hat + self.b_bilinear)[0]
            HO_hat.append(hiO_hat)
            Weights.append(wi)
        HO_hat = dy.concatenate([dy.reshape(ele, d=(1, 2*self.dim_opi)) for ele in HO_hat])
        Weights = dy.concatenate(Weights)
        # length: seq_len
        Weights = dy.softmax(Weights)
        Weights_np = Weights.npvalue()
        ho_summary_t = dy.reshape(Weights, (1, seq_len)) * HO_hat
        return dy.reshape(ho_summary_t, (2*self.dim_opi,)), Weights_np


class ST_dot:
    # selective network with dot attention
    def __init__(self, pc, dim_asp, dim_opi):
        self.pc = pc.add_subcollection()
        self.dim_asp = dim_asp
        self.dim_opi = dim_opi
        self._W_A = self.pc.add_parameters((2*self.dim_opi, 2*self.dim_asp), init=dy.UniformInitializer(0.2))
        self._W_O = self.pc.add_parameters((2*self.dim_opi, 2*self.dim_opi), init=dy.UniformInitializer(0.2))
        self._b = self.pc.add_parameters((2 * self.dim_opi,), init=dy.ConstInitializer(0.0))

    def parametrize(self):
        self.W_A = dy.parameter(self._W_A)
        self.W_O = dy.parameter(self._W_O)
        self.b = dy.parameter(self._b)

    def __call__(self, htA, HO, transform_flag=True):
        """

        :param htA:
        :param HO:
        :param transform_flag: determine if the model needs selective transformation,
        :return:
        """
        seq_len = len(HO)
        HO_hat = []
        Weights = []
        for i in range(seq_len):
            hiO = HO[i]
            if transform_flag:
                hiO_hat = hiO + dy.rectify(self.W_A * htA + self.W_O * hiO + self.b)
            else:
                hiO_hat = hiO
            wi = dy.tanh(dy.dot_product(htA, hiO_hat))
            HO_hat.append(hiO_hat)
            Weights.append(wi)
        HO_hat = dy.concatenate([dy.reshape(ele, d=(1, 2*self.dim_opi)) for ele in HO_hat])
        Weights = dy.concatenate(Weights)
        # length: seq_len
        Weights = dy.softmax(Weights)
        Weights_np = Weights.npvalue()
        ho_summary_t = dy.reshape(Weights, (1, seq_len)) * HO_hat
        return dy.reshape(ho_summary_t, (2*self.dim_opi,)), Weights_np


class ST_concat:
    # selective networks with concat attention
    def __init__(self, pc, dim_asp, dim_opi):
        self.pc = pc.add_subcollection()
        self.dim_asp = dim_asp
        self.dim_opi = dim_opi
        self._W_A = self.pc.add_parameters((2*self.dim_opi, 2*self.dim_asp), init=dy.UniformInitializer(0.2))
        self._W_O = self.pc.add_parameters((2*self.dim_opi, 2*self.dim_opi), init=dy.UniformInitializer(0.2))
        self._b = self.pc.add_parameters((2*self.dim_opi,), init=dy.ConstInitializer(0.0))
        self._W_concat = self.pc.add_parameters((2*self.dim_asp+2*self.dim_opi,), init=dy.UniformInitializer(0.2))

    def parametrize(self):
        self.W_A = dy.parameter(self._W_A)
        self.W_O = dy.parameter(self._W_O)
        self.b = dy.parameter(self._b)
        self.W_concat = dy.parameter(self._W_concat)

    def __call__(self, htA, HO, transform_flag=True):
        """

        :param htA:
        :param HO:
        :param transform_flag: determine if the model needs selective transformation,
        :return:
        """
        seq_len = len(HO)
        HO_hat = []
        Weights = []
        for i in range(seq_len):
            hiO = HO[i]
            if transform_flag:
                hiO_hat = hiO + dy.rectify(self.W_A * htA + self.W_O * hiO + self.b)
            else:
                hiO_hat = hiO
            wi = dy.tanh(dy.dot_product(self.W_concat, dy.concatenate([htA, hiO_hat])))
            HO_hat.append(hiO_hat)
            Weights.append(wi)
        HO_hat = dy.concatenate([dy.reshape(ele, d=(1, 2 * self.dim_opi)) for ele in HO_hat])
        Weights = dy.concatenate(Weights)
        # length: seq_len
        Weights = dy.softmax(Weights)
        Weights_np = Weights.npvalue()
        ho_summary_t = dy.reshape(Weights, (1, seq_len)) * HO_hat
        return dy.reshape(ho_summary_t, (2 * self.dim_opi,)), Weights_np


class BiLSTM:
    # Bi-directional LSTM layer
    def __init__(self, pc, n_in, n_out, dropout_rate, reuse=True):
        self.n_in = n_in
        self.n_out = n_out
        self.dropout_rate = dropout_rate
        self.reuse = reuse
        self.pc = pc
        if self.reuse:
            self.lstm = dy.LSTMBuilder(layers=1, input_dim=self.n_in, hidden_dim=self.n_out, model=self.pc)
        else:
            self.lstm_f = dy.LSTMBuilder(layers=1, input_dim=self.n_in, hidden_dim=self.n_out, model=self.pc)
            self.lstm_b = dy.LSTMBuilder(layers=1, input_dim=self.n_in, hidden_dim=self.n_out, model=self.pc)

    def parametrize(self):
        """
        put the weight matrices onto the computational graph
        :return:
        """
        pass

    def __call__(self, inputs, is_train):
        """

        :param inputs: input word embeddings (a list of expressions), shape: (seq_len, dim_w)
        :return:
        """
        # perform partial dropout
        if is_train:
            dropout_inputs = [dy.dropout(x, self.dropout_rate) for x in inputs]
        else:
            dropout_inputs = [x for x in inputs]
        if self.reuse:
            f_init = self.lstm.initial_state()
            b_init = self.lstm.initial_state()
            H_f = f_init.transduce(dropout_inputs)
            H_b = b_init.transduce(dropout_inputs[::-1])[::-1]
        else:
            f_init = self.lstm_f.initial_state()
            b_init = self.lstm_b.initial_state()
            H_f = f_init.transduce(dropout_inputs)
            H_b = b_init.transduce(dropout_inputs[::-1])[::-1]
        H = [dy.concatenate([f, b]) for (f, b) in zip(H_f, H_b)]
        return H


class BiGRU:
    def __init__(self, pc, n_in, n_out, dropout_rate, reuse=True):
        self.pc = pc
        self.n_in = n_in
        self.n_out = n_out
        self.dropout_rate = dropout_rate
        self.reuse = reuse
        if self.reuse:
            self.gru = dy.GRUBuilder(layers=1, input_dim=self.n_in, hidden_dim=self.n_out, model=self.pc)
        else:
            self.gru_f = dy.GRUBuilder(layers=1, input_dim=self.n_in, hidden_dim=self.n_out, model=self.pc)
            self.gru_b = dy.GRUBuilder(layers=1, input_dim=self.n_in, hidden_dim=self.n_out, model=self.pc)

    def parametrize(self):
        """
        put the weight matrices onto the computational graph
        :return:
        """
        pass

    def __call__(self, inputs, is_train):
        """

        :param inputs: input word embeddings, shape(seq_len, dim_w)
        :param is_train: determine if the partial dropout is needed
        :return:
        """
        if is_train:
            dropout_inputs = [dy.dropout(x, self.dropout_rate) for x in inputs]
        else:
            dropout_inputs = [x for x in inputs]
        if self.reuse:
            f_init = self.gru.initial_state()
            b_init = self.gru.initial_state()
            H_f = f_init.transduce(dropout_inputs)
            H_b = b_init.transduce(dropout_inputs[::-1])[::-1]
        else:
            f_init = self.gru_b.initial_state()
            b_init = self.gru_f.initial_state()
            H_f = f_init.transduce(dropout_inputs)
            H_b = b_init.transduce(dropout_inputs[::-1])[::-1]
        # shape: (seq_len, n * n_out)
        H = [dy.concatenate([f, b]) for (f, b) in zip(H_f, H_b)]
        return H


class Linear:
    # fully connected layer
    def __init__(self, pc, n_in, n_out, use_bias=False):
        self.pc = pc.add_subcollection()
        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = use_bias
        self._W = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        if self.use_bias:
            self._b = self.pc.add_parameters((self.n_out,), init=dy.ConstInitializer(0.0))

    def parametrize(self):
        """

        :return:
        """
        self.W = dy.parameter(self._W)
        if self.use_bias:
            self.b = dy.parameter(self._b)

    def __call__(self, x):
        """

        :param x: input feature vector
        :return:
        """
        Wx = self.W * x
        if self.use_bias:
            Wx = Wx + self.b
        return Wx


class BiLSTMAttention:
    # Note: current implementation only focuses on uni-directional LSTM
    # Note: this class is **not** used in the model
    def __init__(self, pc, n_in, n_out, n_steps, dropout_rate):
        self.pc = pc.add_subcollection()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout_rate = dropout_rate

        # steps in truncated attention
        self.n_steps = n_steps

        self._W = self.pc.add_parameters((4*self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        self._U = self.pc.add_parameters((4*self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._b = self.pc.add_parameters((4*self.n_out), init=dy.ConstInitializer(0.0))

        attention_scale = 1.0 / math.sqrt(1.0) # actually the value is 0.0
        self._u = self.pc.add_parameters((self.n_out,), init=dy.UniformInitializer(attention_scale))

        self._W_h = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._W_x = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        self._W_htilde = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))

    def parametrize(self):
        """

        :return:
        """
        self.W = dy.parameter(self._W)
        self.U = dy.parameter(self._U)
        self.b = dy.parameter(self._b)
        self.u = dy.parameter(self._u)

        self.W_h = dy.parameter(self._W_h)
        self.W_x = dy.parameter(self._W_x)
        self.W_htilde = dy.parameter(self._W_htilde)

    def __call__(self, inputs, is_train=True):
        """
        input word embeddings
        :param inputs:
        :return: a list of hidden states for aspect predictions
        """
        seq_len = len(inputs)
        # hm0 and cm0
        hm = dy.zeros((self.n_steps, self.n_out))
        cm = dy.zeros((self.n_steps, self.n_out))
        h_tilde = dy.zeros((self.n_out,))
        # list of hidden states
        H = []
        for i in range(seq_len):
            xt = inputs[i]
            hm, cm, h_tilde = self.recurrence(xt, hm, cm, h_tilde, dropout_flag=is_train)
            ht = hm[-1]
            H.append(ht)
        return H

    def recurrence(self, xt, hmtm1, cmtm1, h_tilde_tm1, dropout_flag):
        """
        recurrence function of LSTM with truncated self-attention
        :param xt: current input, shape: (n_in)
        :param hmtm1: hidden memory [htm1, ..., h1], shape: (n_steps, n_out)
        :param cmtm1: cell memory: (n_steps, n_out)
        :param h_tilde_tm1: previous hidden summary, shape: (n_out, )
        :param h_tilde_tm1: previous cell summary
        :param dropout_flag: where perform partial dropout
        :return:
        """
        score = dy.concatenate([dy.dot_product(self.u, dy.tanh(\
            self.W_h * hmtm1[i] + self.W_x * xt + self.W_htilde * h_tilde_tm1)) for i in range(self.n_steps)])
        # normalize the attention score
        score = dy.softmax(score)
        # shape: (1, n_out)
        h_tilde_t = dy.reshape(dy.transpose(score) * hmtm1, d=(self.n_out,))
        c_tilde_t = dy.transpose(score) * cmtm1
        Wx = self.W * xt
        if dropout_flag:
            # perform partial dropout over the lstm
            Wx = dy.dropout(Wx, self.dropout_rate)
        Uh = self.U * h_tilde_t
        # shape: (4*n_out)
        sum_item = Wx + Uh + self.b
        it = dy.logistic(sum_item[:self.n_out])
        ft = dy.logistic(sum_item[self.n_out:2*self.n_out])
        ot = dy.logistic(sum_item[2*self.n_out:3*self.n_out])
        c_hat = dy.tanh(sum_item[3*self.n_out:])
        ct = dy.cmult(ft, dy.reshape(c_tilde_t, d=(self.n_out,))) + dy.cmult(it, c_hat)
        ht = dy.cmult(ot, dy.tanh(ct))
        hmt = dy.concatenate([hmtm1[1:], dy.reshape(ht, (1, self.n_out))])
        cmt = dy.concatenate([cmtm1[1:], dy.reshape(ct, (1, self.n_out))])
        return hmt, cmt, h_tilde_t


class BiGRUAttention:
    # By default, we reuse the parameters
    # Note: this class is **not** used in the model
    def __init__(self, pc, n_in, n_out, n_steps, dropout_rate):
        self.pc = pc.add_subcollection()
        self.n_in = n_in
        self.n_out = n_out
        self.n_steps = n_steps
        self.dropout_rate = dropout_rate

        # parameters for recurrent step
        self._W_xr = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        self._W_hr = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._br = self.pc.add_parameters((self.n_out,), init=dy.ConstInitializer(0.0))
        self._W_xz = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        self._W_hz = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._bz = self.pc.add_parameters((self.n_out,), init=dy.ConstInitializer(0.0))
        self._W_xh = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        self._W_hh = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._bh = self.pc.add_parameters((self.n_out,), init=dy.ConstInitializer(0.0))

        # for attention modeling
        attention_scale = 1.0 / math.sqrt(1.0)  # actually the value is 0.0
        self._u = self.pc.add_parameters((self.n_out,), init=dy.UniformInitializer(attention_scale))
        self._W_h = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._W_x = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        self._W_htilde = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))

    def parametrize(self):
        """
        move the weight matrices onto the computational graph
        :return:
        """
        self.W_xr = dy.parameter(self._W_xr)
        self.W_hr = dy.parameter(self._W_hr)
        self.br = dy.parameter(self._br)
        self.W_xz = dy.parameter(self._W_xz)
        self.W_hz = dy.parameter(self._W_hz)
        self.bz = dy.parameter(self._bz)
        self.W_xh = dy.parameter(self._W_xh)
        self.W_hh = dy.parameter(self._W_hh)
        self.bh = dy.parameter(self._bh)

        self.u = dy.parameter(self._u)
        self.W_h = dy.parameter(self._W_h)
        self.W_x = dy.parameter(self._W_x)
        self.W_htilde = dy.parameter(self._W_htilde)

    def __call__(self, inputs, is_train=True):
        """
        forward pass
        :param inputs: input word embeddings
        :return:
        """
        seq_len = len(inputs)
        # hm0
        hm = dy.zeros((self.n_steps+1, self.n_out))
        # h_tilde_0
        h_history = dy.zeros((self.n_out,))
        # list of hidden states
        H = []
        for i in range(seq_len):
            xt = inputs[i]
            hm, h_history = self.recurrence(xt, hm, h_history, dropout_flag=is_train)
            ht = hm[-1]
            H.append(ht)
        return H

    def recurrence(self, xt, hmtm1, h_history_tm1, dropout_flag):
        """

        :param xt: input vector at the time step t
        :param hmtm1: hidden memories in previous n_steps steps
        :param h_tilde_tm1: previous hidden summary
        :param dropout_flag: make a decision for conducting partial dropout
        :return:
        """
        score = dy.concatenate([dy.dot_product(self.u, dy.tanh( \
            self.W_h * hmtm1[i] + self.W_x * xt + self.W_htilde * h_history_tm1)) for i in range(self.n_steps)])
        # normalize the attention score
        score = dy.softmax(score)
        # shape: (1, n_out), history of [h[t-n_steps-1], ..., h[t-2]]
        h_history_t = dy.reshape(dy.transpose(score) * hmtm1[:-1], d=(self.n_out,))
        htm1 = hmtm1[-1]
        #h_tilde_t = dy.concatenate([h_history_t, htm1])
        h_tilde_t = htm1 + dy.rectify(h_history_t)
        if dropout_flag:
            # perform partial dropout, i.e., add dropout over the matrices W_x*
            rt = dy.logistic(dy.dropout(self.W_xr, self.dropout_rate) * xt + self.W_hr * h_tilde_t + self.br)
            zt = dy.logistic(dy.dropout(self.W_xz, self.dropout_rate) * xt + self.W_hz * h_tilde_t + self.bz)
            ht_hat = dy.tanh(dy.dropout(self.W_xh, self.dropout_rate) * xt + self.W_hh * dy.cmult(rt, h_tilde_t) \
                             + self.bh)
            ht = dy.cmult(zt, h_tilde_t) + dy.cmult((1.0 - zt), ht_hat)
        else:
            rt = dy.logistic(self.W_xr * xt + self.W_hr * h_tilde_t + self.br)
            zt = dy.logistic(self.W_xz * xt + self.W_hz * h_tilde_t + self.bz)
            ht_hat = dy.tanh(self.W_xh * xt + self.W_hh * dy.cmult(rt, h_tilde_t) + self.bh)
            ht = dy.cmult(zt, h_tilde_t) + dy.cmult((1.0 - zt), ht_hat)
        hmt = dy.concatenate([hmtm1[1:], dy.reshape(ht, (1, self.n_out))])
        return hmt, h_history_t


class Model:
    def __init__(self, params, vocab, label2tag, pretrained_embeddings=None):
        """

        :param params:
        :param vocab:
        :param label2tag:
        :param pretrained_embeddings:
        """
        self.dim_w = params.dim_w
        self.win = params.win
        self.vocab = vocab
        self.n_words = len(self.vocab)
        self.dim_asp = params.dim_asp
        self.dim_opi = params.dim_opi
        self.dim_y_asp = params.n_asp_tags
        self.dim_y_opi = params.n_opi_tags
        self.n_steps = params.n_steps
        self.asp_label2tag = label2tag
        self.opi_label2tag = {0: 'O', 1: 'T'}
        self.dropout_asp = params.dropout_asp
        self.dropout_opi = params.dropout_opi
        self.dropout = params.dropout
        self.rnn_type = params.rnn_type
        self.ds_name = params.ds_name
        self.model_name = params.model_name
        self.attention_type = params.attention_type

        self.pc = dy.ParameterCollection()
        self.Emb = WDEmb(pc=self.pc, n_words=self.n_words, dim_w=self.dim_w,
                         pretrained_embeddings=pretrained_embeddings)
        #self.ASP_RNN = LSTM(pc=self.pc, n_in=self.win*self.dim_w, n_out=self.dim_asp, dropout_rate=self.dropout_asp)
        #self.OPI_RNN = LSTM(pc=self.pc, n_in=self.win*self.dim_w, n_out=self.dim_opi, dropout_rate=self.dropout_opi)
        # use dynet RNNBuilder rather than the self-defined RNN classes
        if self.rnn_type == 'LSTM':
            self.ASP_RNN = dy.LSTMBuilder(1, self.win * self.dim_w, self.dim_asp, self.pc)
            self.OPI_RNN = dy.LSTMBuilder(1, self.win * self.dim_w, self.dim_opi, self.pc)
        elif self.rnn_type == 'GRU':
            # NOT TRIED!
            self.ASP_RNN = dy.GRUBuilder(1, self.win * self.dim_w, self.dim_asp, self.pc)
            self.OPI_RNN = dy.GRUBuilder(1, self.win * self.dim_w, self.dim_opi, self.pc)
        else:
            raise Exception("Invalid RNN type!!!")
        self.THA = THA(pc=self.pc, n_steps=self.n_steps, n_in=2*self.dim_asp)
        if self.attention_type == 'bilinear':
            self.STN = ST_bilinear(pc=self.pc, dim_asp=self.dim_asp, dim_opi=self.dim_opi)
        # here dot attention is not applicable since the aspect representation and opinion representation
        # have different dimensions
        # elif self.attention_type == 'dot':
        #    self.STN = ST_dot(pc=self.pc, dim_asp=self.dim_asp, dim_opi=self.dim_opi)
        elif self.attention_type == 'concat':
            self.STN = ST_concat(pc=self.pc, dim_asp=self.dim_asp, dim_opi=self.dim_opi)
        else:
            raise Exception("Invalid attention type!!!")

        self.ASP_FC = Linear(pc=self.pc, n_in=2*self.dim_asp+2*self.dim_opi, n_out=self.dim_y_asp)
        self.OPI_FC = Linear(pc=self.pc, n_in=2*self.dim_opi, n_out=self.dim_y_opi)

        self.layers = [self.ASP_FC, self.OPI_FC, self.THA, self.STN]

        if params.optimizer == 'sgd':
            self.optimizer = dy.SimpleSGDTrainer(self.pc, params.sgd_lr)
        elif params.optimizer == 'momentum':
            self.optimizer = dy.MomentumSGDTrainer(self.pc, 0.01, 0.9)
        elif params.optimizer == 'adam':
            self.optimizer = dy.AdamTrainer(self.pc, 0.001, 0.9, 0.9)
        elif params.optimizer == 'adagrad':
            self.optimizer = dy.AdagradTrainer(self.pc)
        elif params.optimizer == 'adadelta':
             # use default value of adadelta
            self.optimizer = dy.AdadeltaTrainer(self.pc)
        else:
            raise Exception("Invalid optimizer!!")

    def parametrize(self):
        """

        :return:
        """
        for layer in self.layers:
            layer.parametrize()

    def __call__(self, dataset, is_train=True):
        """

        :param dataset: input dataset
        :param is_train: train flag
        :return:
        """
        n_samples = len(dataset)
        total_loss = 0.0
        Y_pred_asp, Y_pred_opi = [], []

        opinion_attention_outputs = []
        aspect_attention_outputs = []
        time_costs = []
        for i in range(n_samples):
            beg = time.time()
            dy.renew_cg()
            self.parametrize()

            self.words = dataset[i]['wids']
            self.y_asp = dataset[i]['labels']
            self.y_opi = dataset[i]['opinion_labels']
            raw_words = dataset[i]['words']

            input_embeddings = self.Emb(xs=self.words)

            f_asp = self.ASP_RNN.initial_state()
            b_asp = self.ASP_RNN.initial_state()
            f_opi = self.OPI_RNN.initial_state()
            b_opi = self.OPI_RNN.initial_state()

            # these operations are equivalent to partial dropout in LSTM
            if is_train:
                input_asp = [dy.dropout(x, self.dropout_asp) for x in input_embeddings]
                input_opi = [dy.dropout(x, self.dropout_opi) for x in input_embeddings]
            else:
                input_asp = input_embeddings
                input_opi = input_embeddings
            H_asp_f = f_asp.transduce(input_asp)
            H_asp_b = b_asp.transduce(input_asp[::-1])[::-1]

            H_asp = [dy.concatenate([f, b]) for (f, b) in zip(H_asp_f, H_asp_b)]
            H_opi_f = f_opi.transduce(input_opi)
            H_opi_b = b_opi.transduce(input_opi[::-1])[::-1]

            H_opi = [dy.concatenate([f, b]) for (f, b) in zip(H_opi_b, H_opi_f)]
            asp_predictions, opi_predictions = [], []
            seq_len = len(input_embeddings)
            # aspect representations encoding history information
            H_asp_tilde, asp_attentions = self.THA(H=H_asp)
            assert len(asp_attentions) == len(raw_words)

            losses = []
            # a collection of opinion summary
            H_opi_summ = []
            for t in range(seq_len):
                # got the original words from the dataset
                #wt = raw_words[t]
                #current_line = wt + ":"
                htA = H_asp_tilde[t]
                # opinion summary at the time step t
                summary, weights = self.STN(htA=htA, HO=H_opi, transform_flag=True)

                H_opi_summ.append(summary)
                asp_feat = dy.concatenate([htA, summary])
                opi_feat = H_opi[t]
                if is_train:
                    # in the training phase, perform dropout
                    asp_feat = dy.dropout(asp_feat, self.dropout)
                    opi_feat = dy.dropout(opi_feat, self.dropout)
                p_y_x_asp = self.ASP_FC(x=asp_feat)
                p_y_x_opi = self.OPI_FC(x=opi_feat)
                asp_predictions.append(p_y_x_asp.npvalue())
                opi_predictions.append(p_y_x_opi.npvalue())

                target_asp = self.y_asp[t]
                target_opi = self.y_opi[t]
                loss_asp = dy.pickneglogsoftmax(p_y_x_asp, target_asp)
                loss_opi = dy.pickneglogsoftmax(p_y_x_opi, target_opi)

                losses.append(loss_asp + loss_opi)
            end = time.time()
            time_cost = end - beg
            #print("time cost of current sample:", time_cost)
            time_costs.append(time_cost)
            # add a split token after each sentence
            #opinion_attention_outputs.append("\n")
            loss = dy.esum(losses)
            total_loss += loss.scalar_value()
            if is_train:
                loss.backward()
                self.optimizer.update()
            pred_asp_labels = np.argmax(np.array(asp_predictions), axis=1)
            pred_opi_labels = np.argmax(np.array(opi_predictions), axis=1)
            pred_asp_tags = bio2ot(tag_sequence=[self.asp_label2tag[l] for l in pred_asp_labels])
            pred_opi_tags = [self.opi_label2tag[l] for l in pred_opi_labels]
            Y_pred_asp.append(pred_asp_tags)
            Y_pred_opi.append(pred_opi_tags)
        #print("Average time costs:", sum(time_costs) / len(time_costs))
        return total_loss, Y_pred_asp, Y_pred_opi, aspect_attention_outputs, opinion_attention_outputs
