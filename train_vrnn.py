import numpy as np
import tensorflow as tf
import argparse
import time
from datetime import datetime
import os
from network import Network
from cmpassDataloader import dataLoader
from matplotlib import pyplot as plt

class trainer():
    def __init__(self, args, model):
        self.args = args
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.resume = args.resume
        if self.resume:
            self.dirname = args.resume_path
            # self.logs = args.log_path
        else:
            self.dirname = 'save-vrnn/' + datetime.now().isoformat().replace(':', '-')
        self.logs = 'logs/' + datetime.now().isoformat().replace(':', '-')
        self.model = model
        self.picpath = args.pic_path + '/'+  datetime.now().isoformat().replace(':', '-')
        if not os.path.exists(self.picpath):
            os.makedirs(self.picpath)

    # def sample_check(self, x, sess, k):
    #     x_rec, mu, sigma = sess.run([self.model.rho, self.model.enc_mu, self.model.enc_sigma], {model.input_data: x}) # mu shape (batch,  seq, 1)
    #     x_rec = np.reshape(x_rec, [self.batch_size, self.seq_length, -1])
    #     x = np.reshape(x, [self.batch_size, self.seq_length, -1])
    #     mu = np.reshape(mu, [self.batch_size, self.seq_length, -1])
    #     sigma = np.reshape(sigma, [self.batch_size, self.seq_length, -1])
    #     x1 = np.reshape(x[1, :, :], [self.seq_length, -1])
    #     x_rec1 = np.reshape(x_rec[1, :, :], [self.seq_length, -1])

    #     mu1 = np.reshape(mu[1, :, :], [self.seq_length, -1])
    #     sigma1 = np.reshape(sigma[1,:,:], [self.seq_length, -1])
    #     plt.figure()
    #     ax1 = plt.subplot(411)
    #     plt.plot(range(self.seq_length), x1[:,1])

    #     ax2 = plt.subplot(412)
    #     plt.plot(range(self.seq_length), x1[:,6])

    #     ax3 = plt.subplot(413)
    #     plt.plot(range(self.seq_length), x_rec1[:, 1])

    #     ax4 = plt.subplot(414)
    #     plt.plot(range(self.seq_length), x_rec1[:, 6])
    #     ax1.set_title("x")
    #     ax3.set_title("x_rec")
    #     plt.savefig(self.picpath+'/'+'rec'+str(k)+'.png')
    #     plt.close()


    #     plt.figure()
    #     ax1 = plt.subplot(211)
    #     plt.plot(range(self.seq_length), mu1)
    #     ax2 = plt.subplot(212)
    #     plt.plot(range(self.seq_length), sigma1)
    #     plt.tight_layout()
    #     # plt.show()
    #     ax1.set_title("u_sample")
    #     ax2.set_title("sigma_sample")
    #     plt.savefig(self.picpath+'/'+str(k)+'.png')
    #     plt.close()
    #     # plt.show()

    # def imshow(x):
    #     x1 = np.reshape(x[1, :, :], [args.seq_length, -1])
    #     plt.figure()
    #     ax1 = plt.subplot(211)
    #     plt.plot(range(args.seq_length), x1[:,1])
    #     ax2 = plt.subplot(212)
    #     plt.plot(range(args.seq_length), x1[:,6])
    #     plt.show()



    def train(self):
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        ckpt = tf.train.get_checkpoint_state(self.dirname)
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(self.logs, sess.graph)
            check = tf.add_check_numerics_ops()
            merged = tf.summary.merge_all()
            # if not args.resume:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print ("Loaded model")
            start = time.time()
            data = dataLoader(bs = self.batch_size, sl = args.seq_length)
            n_batches = data.batches
            for e in range(args.num_epochs):
                sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e))) #把lr设置为variable,然后用动态变化
                # state = model.initial_state_c, model.initial_state_h

                for b in range(n_batches):
                    x, y = data.nextBatch()
                    feed = {model.input_data: x, model.target_data: y}
                    kl,li,train_loss,  _, cr, summary, sigma, mu,  target= sess.run(
                            [model.kl_loss, model.li_loss,model.lossfunc, model.train_op, check, merged, model.enc_sigma, model.enc_mu, model.rulout ],
                                                                 feed)
                    summary_writer.add_summary(summary, e * n_batches + b)
                    if (e * n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                        checkpoint_path = os.path.join(self.dirname, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=e * n_batches + b)
                        print ("model saved to {}".format(checkpoint_path))
                
                    end = time.time()
                    print("{}/{} (epoch {}), train_loss = {:.6f},time/batch = {:.1f}, std = {:.3f}, mu = {:.3f},kl = {:.6f},li = {:.3f}" \
                        .format(e * n_batches + b,
                                args.num_epochs * n_batches,
                                e, train_loss, end - start, np.mean(sigma), np.mean(mu),kl, li))
                    start = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=100,
                        help='size of RNN hidden state')
    parser.add_argument('--latent_size', type=int, default=200,
                        help='size of latent space')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='minibatch size')
    parser.add_argument('--resume',type = bool , default = False,
                        help = 'continue')
    parser.add_argument('--resume_path',type =str, default = './save-vrnn/2019-04-28T22-30-06.880834',help = 'the path to continue' )
    parser.add_argument('--pic_path', type=str, default='./picture',
                        help='mu increase')
    parser.add_argument('--log_path', type = str)
    parser.add_argument('--seq_length', type=int, default=10,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--input_size', type=int, default=18,
                        help='input_size')
    parser.add_argument('--output_size', type=int, default=1,
                        help='output_size')

    args = parser.parse_args()
    model = Network(args)
    T = trainer(args, model)
    T.train()
