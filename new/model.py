import tensorflow as tf
from utils import *
import numpy as np
import time
def conv_relu(input,kernel_shape,bias_shape):
    with tf.variable_scope('block'):
    	output = tf.layers.conv2d(input, kernel_shape[3], kernel_shape[0], padding='same', activation=tf.nn.relu)
    return output
def cnn1(input,size_channel):
    with tf.variable_scope('conv1'):
        out1 = conv_relu(input,[3,3,size_channel,64],[64])
    with tf.variable_scope('conv2'):
        out2 = conv_relu(out1,[3,3,64,64],[64])
    with tf.variable_scope('conv3'):
        out3 = conv_relu(out2,[3,3,64,64],[64])
    with tf.variable_scope('conv4'):
        out4 = conv_relu(out3,[3,3,64,128],[128])
        out4 = tf.nn.max_pool(out4,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    return out4
def cnn2(input,size_channel,size_m):
    with tf.variable_scope('conv1'):
        out1 = conv_relu(input,[3,3,size_channel,128],[128])
    with tf.variable_scope('conv2'):
        out2 = conv_relu(out1,[3,3,128,128],[128])
    with tf.variable_scope('conv3'):
        out3 = conv_relu(out2,[3,3,128,128],[128])
    with tf.variable_scope('conv4'):
        out4 = conv_relu(out3,[3,3,128,512],[512])
        out4 = tf.image.resize_images(out4,(2*size_m,2*size_m),method=0)
    return out4
def dcnn(input,size_channel,size_m):
    with tf.variable_scope('cnn11'):
        out11 = cnn1(input,size_channel)
    with tf.variable_scope('cnn12'):
        out12 = cnn1(out11,128)
    with tf.variable_scope('cnn13'):
        out13 = cnn1(out12,128)
    with tf.variable_scope('cnn14'):
        out14 = cnn1(out13,128)
    with tf.variable_scope('cnn24'):
        out24 = cnn2(out14,128,size_m)
        out24 = tf.image.resize_images(out24,(size_m,size_m),method=0)
        out24 = tf.concat([out13,out13,out13,out13],axis = 3) + out24
    with tf.variable_scope('cnn23'):
        out23 = cnn2(out24,512,size_m)
        out23 = tf.image.resize_images(out23,(size_m*2,size_m*2),method=0)
        out23 = out23 + tf.concat([out12,out12,out12,out12],axis = 3)
    with tf.variable_scope('cnn22'):
        out22 = cnn2(out23,512,size_m)
        out22 = tf.image.resize_images(out22,(size_m*4,size_m*4),method=0)
        out22 = out22 + tf.concat([out11,out11,out11,out11],axis = 3)
    with tf.variable_scope('cnn21'):
        out21 = cnn2(out22,512,size_m)
        out21 = tf.image.resize_images(out22,(size_m*8,size_m*8),method=0)
    with tf.variable_scope('cnn3'):
        out = conv_relu(out21,[3,3,512,size_channel],[size_channel])
        output = out + input
    return output
def module(input, is_training=True, out_channels=1,kstage=1,size_m = 32):
    delta1 = tf.get_variable(name='d1',shape=[1],initializer=tf.ones_initializer())     
    x1 = tf.multiply(delta1,input)
    x2 = x1
    with tf.variable_scope("denoise_cnn") as scope:
        for i in range (kstage):
            delta2 = tf.get_variable(name='d2%d' %i,shape=[1],initializer=tf.ones_initializer())
            delta3 = tf.get_variable(name='d3%d' %i,shape=[1],initializer=tf.ones_initializer())
            x2 = dcnn(x2,out_channels,size_m)
            x2 = tf.multiply(delta3,input) + tf.multiply(delta2,x2)
            scope.reuse_variables()
        x2 = dcnn(x2,out_channels,size_m)
    return x2
    
class denoiser(object):
    def __init__(self,sess,input_c_dim=1,sigma=25,batch_size=128,kstage=1):
        self.sess = sess
        self.sigma = sigma
        self.input_c_dim = input_c_dim
        self.kstage = kstage
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.size_m = tf.placeholder(tf.int32, name='size_m')
        self.X = self.Y_ + tf.random_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images
        # self.X = self.Y_ + tf.truncated_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images
        self.Y = module(self.X, is_training=self.is_training,out_channels=self.input_c_dim,kstage=1,size_m = self.size_m)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.losses1 = tf.reduce_max(self.Y_)
        self.losses2 = tf.reduce_max(self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr,name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, test_data, sample_dir, summary_merged, summary_writer):
        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        psnr_sum = 0
        for idx in range(len(test_data)):
            clean_image = test_data[idx].astype(np.float32) / 255.0
            output_clean_image, noisy_image, psnr_summary = self.sess.run(
                [self.Y, self.X, summary_merged],
                feed_dict={self.Y_: clean_image,
                           self.is_training: False, self.size_m:clean_image.shape[1]//8})
            summary_writer.add_summary(psnr_summary, iter_num)
            groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx + 1, psnr))
            psnr_sum += psnr
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                        groundtruth, noisyimage, outputimage)
	    
        avg_psnr = psnr_sum / len(test_data)

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

    def denoise(self, data):
        output_clean_image, noisy_image, psnr = self.sess.run([self.Y, self.X, self.eva_psnr],
                                                              feed_dict={self.Y_: data,
                                                                         self.is_training: False,
                                                                         self.size_m:32})
        return output_clean_image, noisy_image, psnr

    def train(self, data, eval_data, batch_size, ckpt_dir, epoch, lr, sample_dir, eval_every_epoch=1):
        # assert data range is between 0 and 1
        numBatch = int(data.shape[0] / batch_size)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                      summary_writer=writer)  # eval_data value range is 0-255
        for epoch in range(start_epoch, epoch):
            np.random.shuffle(data)
            for batch_id in range(start_step, numBatch):
                batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                batch_images = batch_images.astype(np.float32) / 255.0 # normalize the data to 0-1
                _, losses1,losses2,loss, summary = self.sess.run([self.train_op,self.losses1, self.losses2,self.loss, merged],
                                                 feed_dict={self.Y_: batch_images, self.lr: lr[epoch],
                                                            self.is_training: True, self.size_m:5})
                #print(losses1)
                #print(losses2)
                #print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                #        % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                if(np.mod(batch_id + 1 , 10) == 0):		#print every 10 batch
                    sys.stdout.write("\rEpoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                                        % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                    sys.stdout.flush()
                if(np.mod(batch_id + 1,200) == 0 or (batch_id) + 1 == numBatch):	#change every 200 batch
                    print('')
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                              summary_writer=writer)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")
        
    def save(self, iter_num, ckpt_dir, model_name='DnCNN-denoising'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)
        

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0
        

    def test(self, test_files, ckpt_dir, save_dir):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        for idx in range(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image, self.is_training: False, self.size_m:32})
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(save_dir, 'noisy%d.png' % idx), noisyimage)
            save_images(os.path.join(save_dir, 'denoised%d.png' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
        
        
