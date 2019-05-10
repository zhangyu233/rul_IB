import numpy as np
import tensorflow as tf
from vrnn_cell import VartiationalRNNCell


class Network():
    def __init__(self, args, sample=False):
        self.args = args
        cell = VartiationalRNNCell(args.input_size, args.rnn_size, args.latent_size)
        self.cell = cell
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, args.input_size], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.output_size],name = 'target_data')
        self.initial_state_c, self.initial_state_h = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # input shape: (batch_size, n_steps, n_input)
        with tf.variable_scope("inputs"):
            inputs = tf.transpose(self.input_data, [1, 0, 2])  
            inputs = tf.reshape(inputs, [-1, args.input_size]) # (n_steps*batch_size, n_input)
            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            inputs = tf.split(axis=0, num_or_size_splits=args.seq_length, value=inputs) # n_steps * (batch_size, n_hidden) list中有n_steps个(batch_size, n_hidden）

        self.input = tf.stack(inputs)#(n_steps, batch_size, n_hidden)
        # Get vrnn cell output
        outputs, last_state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=(self.initial_state_c, self.initial_state_h))
        #outputs : seqlength的list
        #  每个里面是个tuple，（mu, sigma,rho)
        #mu dim (60 ,100) (60, 100） ....
        outputs_reshape = []
        names = ["enc_mu", "enc_sigma",  "dec_rho"]
        for n, name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])# (length, batch_size , ndim)
                x = tf.transpose(x,[1,0,2])#(batch_size ,length , n_dim)
                outputs_reshape.append(x)

        enc_mu, enc_sigma, dec_rho = outputs_reshape
        self.rulout = tf.reshape(dec_rho[:,-1,:],[args.batch_size, -1])

        def tf_normal(y,  rho):
            with tf.variable_scope('normal'):
                result = tf.reduce_mean(tf.square(rho - y))
                return result


        def tf_kl_istropic(mu_1, sigma_1):
            with tf.variable_scope("kl_gaussI"):
                return  0.5 * tf.reduce_sum(tf.square(sigma_1)+ tf.square(mu_1) - 1. -2 * tf.log(tf.maximum(1e-9,sigma_1), name = 'log_sigma_1'), 2)



        self.kl_loss = tf.reduce_mean(tf_kl_istropic(enc_mu, enc_sigma))
        self.li_loss = tf_normal(self.rulout, self.target_data )
        self.enc_mu = enc_mu# (b, l ,d)
        self.enc_sigma = enc_sigma       
        self.final_state_c, self.final_state_h = last_state
        self.lossfunc = self.kl_loss + self.li_loss

        tf.summary.scalar('cost', self.lossfunc)
        tf.summary.scalar('kl_cost', self.kl_loss)
        tf.summary.scalar('c_cost',self.li_loss)
        tf.summary.scalar('mu', tf.reduce_mean(self.enc_mu)) # enc_mu1 shape is (seq_length, batch_size ,1)
        tf.summary.scalar('sigma', tf.reduce_mean(self.enc_sigma))

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        for t in tvars:
            print (t.name)

        grads = tf.gradients(self.lossfunc, tvars)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
