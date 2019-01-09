import tensorflow as tf
from utils import *
import numpy as np
import time
def conv_relu(input,filternum,kernel_size,is_training=True,batch_normalize=False):
    with tf.variable_scope('block'):
        if(batch_normalize):
            output = tf.layers.conv2d(input, filternum, kernel_size, padding='same')
            output = tf.nn.relu(output)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        else:
            output = tf.layers.conv2d(input, filternum, kernel_size, padding='same',activation=tf.nn.relu)
    return output
def up_conv(input,numfilters):
    with tf.variable_scope('up_conv_block'):
        output = tf.layers.conv2d_transpose(input,filters = numfilters,kernel_size= (2,2),strides= (2,2),padding= 'valid',activation= tf.nn.relu,)
    return output
def cnn1(input,size_channel,is_training):
    with tf.variable_scope('conv1'):
        out1 = conv_relu(input,64,3,is_training)
    with tf.variable_scope('conv2'):
        out2 = conv_relu(out1,64,3,is_training)
    with tf.variable_scope('conv3'):
        out3 = conv_relu(out2,64,3,is_training)
    with tf.variable_scope('conv4'):
        out4 = conv_relu(out3,128,3,is_training)
        out4 = tf.nn.max_pool(out4,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    return out4
def cnn2(input,size_channel,is_training):
    with tf.variable_scope('conv1'):
        out1 = conv_relu(input,128,3,is_training)
    with tf.variable_scope('conv2'):
        out2 = conv_relu(out1,128,3,is_training)
    with tf.variable_scope('conv3'):
        out3 = conv_relu(out2,128,3,is_training)
    with tf.variable_scope('conv4'):
        out4 = conv_relu(out3,512,3,is_training)
    return out4
def dcnn(input,size_channel,size_m,is_training):
    with tf.variable_scope('cnn11'):
        out11 = cnn1(input,size_channel,is_training)
    with tf.variable_scope('cnn12'):
        out12 = cnn1(out11,128,is_training)
    with tf.variable_scope('cnn13'):
        out13 = cnn1(out12,128,is_training)
    with tf.variable_scope('cnn14'):
        out14 = cnn1(out13,128,is_training)
    with tf.variable_scope('cnn24'):
        out24 = cnn2(out14,128,is_training)
        out24 = up_conv(out24,512)
        #out24 = tf.image.resize_images(out24,(size_m,size_m),method=0)
        out24 = tf.concat([out13,out13,out13,out13],axis = 3) + out24
    with tf.variable_scope('cnn23'):
        out23 = cnn2(out24,512,is_training)
        out23 = up_conv(out23,512)
        #out23 = tf.image.resize_images(out23,(size_m*2,size_m*2),method=0)
        out23 = out23 + tf.concat([out12,out12,out12,out12],axis = 3)
    with tf.variable_scope('cnn22'):
        out22 = cnn2(out23,512,is_training)
        out22 = up_conv(out22,512)
        #out22 = tf.image.resize_images(out22,(size_m*4,size_m*4),method=0)
        out22 = out22 + tf.concat([out11,out11,out11,out11],axis = 3)
    with tf.variable_scope('cnn21'):
        out21 = cnn2(out22,512,is_training)
        out21 = up_conv(out21,512)
        #out21 = tf.image.resize_images(out22,(size_m*8,size_m*8),method=0)
    with tf.variable_scope('cnn3'):
        out = conv_relu(out21,size_channel,3,is_training)
        output = out + input
    return output
def module(input, is_training=True, out_channels=1,kstage=1,size_m = 32):
    delta1 = tf.get_variable(name='d1',shape=[1],initializer=tf.constant_initializer(0))     
    x1 = tf.multiply(delta1,input)
    x_in = x1
    print("kstage = %d"%kstage)
    out = []

    with tf.variable_scope("denoise_cnn") as scope:
        for stage in range(0,kstage + 1):
            delta2 = tf.get_variable(name='d20_%d'%stage, shape=[1], initializer=tf.constant_initializer(0))
            delta3 = tf.get_variable(name='d30_%d'%stage, shape=[1], initializer=tf.constant_initializer(0))
            with tf.variable_scope("", reuse=(False or stage>0)):
                x_out = dcnn(x_in, out_channels, size_m, is_training)
            x_in = tf.multiply(delta3, input) + tf.multiply(delta2, x_out)
            out.append(x_out)
    return out


'''
    with tf.variable_scope("denoise_cnn") as scope:
        delta2 = tf.get_variable(name='d20',shape=[1],initializer=tf.constant_initializer(0))
        delta3 = tf.get_variable(name='d30',shape=[1],initializer=tf.constant_initializer(0))
        with tf.variable_scope("",reuse=False):
            x2 = dcnn(x2,out_channels,size_m,is_training)
        x3 = tf.multiply(delta3,input) + tf.multiply(delta2,x2)
        with tf.variable_scope("",reuse=True):
            x3 = dcnn(x3,out_channels,size_m,is_training)
        delta2 = tf.get_variable(name='d21',shape=[1],initializer=tf.constant_initializer(0))
        delta3 = tf.get_variable(name='d31',shape=[1],initializer=tf.constant_initializer(0))
        x4 = tf.multiply(delta3,input) + tf.multiply(delta2,x3)
        with tf.variable_scope("",reuse=True):
            x4 = dcnn(x4,out_channels,size_m,is_training)
        delta2 = tf.get_variable(name='d22',shape=[1],initializer=tf.constant_initializer(0))
        delta3 = tf.get_variable(name='d32',shape=[1],initializer=tf.constant_initializer(0))
        x5 = tf.multiply(delta3,input) + tf.multiply(delta2,x4)
        with tf.variable_scope("",reuse=True):
            x5 = dcnn(x5,out_channels,size_m,is_training)



        for i in range (kstage):
            delta2 = tf.get_variable(name='d2%d' %i,shape=[1],initializer=tf.constant_initializer(0))
            delta3 = tf.get_variable(name='d3%d' %i,shape=[1],initializer=tf.constant_initializer(0))
            if(i != 0):
                with tf.variable_scope("",reuse=True):
                    x2 = dcnn(x2,out_channels,size_m,is_training)
            else:
                with tf.variable_scope("",reuse=False):
                    x2 = dcnn(x2,out_channels,size_m,is_training)
            x3 = tf.multiply(delta3,input) + tf.multiply(delta2,x2)
            #scope.reuse_variables()
        if(kstage != 0):
            with tf.variable_scope("",reuse=True):
                x3 = dcnn(x3,out_channels,size_m,is_training)
        else:
            with tf.variable_scope("",reuse=False):
                x3 = dcnn(x3,out_channels,size_m,is_training)
    return x3,x2
'''
    
class denoiser(object):
    def __init__(self,sess,input_c_dim=1,sigma=25,batch_size=128,kstage=1):
        self.sess = sess
        self.sigma = sigma
        self.input_c_dim = input_c_dim
        self.kstage = kstage
        self.batch_size = batch_size
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.size_m = tf.placeholder(tf.int32, name='size_m')
        self.X = self.Y_ + tf.random_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images
        # self.X = self.Y_ + tf.truncated_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images

        self.Y_list = module(self.X, is_training=self.is_training,out_channels=self.input_c_dim,kstage=self.kstage,size_m = self.size_m)
        #self.Y,self.Y2,self.Y3,self.Y4 = module(self.X, is_training=self.is_training,out_channels=self.input_c_dim,kstage=3,size_m = self.size_m)
        
        self.loss_group = self.calculate_loss_group() 
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y_list, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr,name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_group = []
        with tf.control_dependencies(update_ops):
            for i in range(0,self.kstage+1):
                self.train_group.append(optimizer.minimize(self.loss_group[i]))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")
    def calculate_loss_group(self):
        loss_group = []
        for i in range(0,self.kstage+1):
            loss_group.append((1.0 / self.batch_size) * tf.nn.l2_loss(self.Y_ - self.Y_list[i]))
        return loss_group
    def evaluate(self, iter_num, test_data, sample_dir,select_num,Show_final = True):
        # assert test_data value range is 0-255
        if Show_final:
            print("[*] Evaluating...")
        psnr_sum = 0
        for idx in range(len(test_data)):
            clean_image = test_data[idx].astype(np.float32) / 255.0
            output_clean_image, noisy_image = self.sess.run(
                [self.Y_list[select_num], self.X],
                feed_dict={self.Y_: clean_image,
                           self.is_training: False, self.size_m:clean_image.shape[1]//8})
            groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            if Show_final:
                print("img%d PSNR: %.2f" % (idx + 1, psnr))
            psnr_sum += psnr
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                        groundtruth, noisyimage, outputimage)
	    
        avg_psnr = psnr_sum / len(test_data)
        if Show_final:
            print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
        return avg_psnr
    def denoise(self, data):
        output_clean_image, noisy_image, psnr = self.sess.run([self.Y, self.X, self.eva_psnr],
                                                              feed_dict={self.Y_: data,
                                                                         self.is_training: False,
                                                                         self.size_m:32})
        return output_clean_image, noisy_image, psnr
    def train_select(self,select_num,batch_images,lr):

        _, loss = self.sess.run([self.train_group[select_num], self.loss_group[select_num]],
                                         feed_dict={self.Y_: batch_images, self.lr: lr,
                                                    self.is_training: True, self.size_m: 6})
        return loss
    def train(self, data, eval_data, batch_size, ckpt_dir, epoch, lr_init, lr_decay, sample_dir, config, eval_every_epoch=1):
        # assert data range is between 0 and 1
        numBatch = int(data.shape[0] / batch_size)
        # load pretrained model
        #load_model_status, global_step = self.load(ckpt_dir)
        #if load_model_status:
        #    iter_num = global_step
        #    start_epoch = global_step // numBatch
        #    start_step = global_step % numBatch
        #    print("[*] Model restore success!")
        #else:
        lrs = [ld*lr_init for ld in lr_decay]
	iter_num = 0
        start_epoch = 0
        start_step = 0
        print("[*] Not find pretrained model!")


        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        stage = 0
        self.evaluate(iter_num, eval_data, sample_dir=sample_dir,select_num=stage)  # eval_data value range is 0-255
        ttl = epoch
        pnsr_max = 0
	pnsr_hist_max = 0
        psnr = 0
	index = 0
        pnsr_list = []
        for epoch in range(start_epoch, epoch):
            np.random.shuffle(data)
            for batch_id in range(start_step, numBatch):
		lr = lrs[index]
                batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                batch_images = batch_images.astype(np.float32) / 255.0 # normalize the data to 0-1

                loss = self.train_select(stage, batch_images, lr)

                if(np.mod(batch_id + 1 , 50) == 0):
                    psnr = self.evaluate(iter_num, eval_data, sample_dir=sample_dir,  select_num=stage, Show_final=False)
                    if pnsr_max < psnr:
                        pnsr_max = psnr
		    if pnsr_hist_max < psnr:
			pnsr_hist_max = psnr
                    pnsr_list.append(pnsr_max)
		max_stat = 30*config.stay_epoch
                if len(pnsr_list) > max_stat:
                    if pnsr_max == pnsr_list[-max_stat]:
			if index < len(lrs)-1:
			    index += 1
			else:
			    index = 0
                            stage += 1
			    pnsr_max = 0 
			pnsr_list = []
			if stage > self.kstage:
			    self.evaluate(iter_num, eval_data, sample_dir=sample_dir,select_num=stage-1)
			    break
			continue
                if(np.mod(batch_id + 1 , 10) == 0):		#print every 10 batch
                    sys.stdout.write("\rStage: [%d] Epoch: [%2d] [%4d/%4d] time: %4.4f,lr: %f, loss: %.6f, psnr: %.2f, Max_psnr: %.2f, Max_hist_psnr: %.2f."
                                        % (stage, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, lr, loss, psnr, pnsr_max, pnsr_hist_max))
                    sys.stdout.flush()
                if(np.mod(batch_id + 1,200) == 0 or (batch_id) + 1 == numBatch):	#change every 200 batch
                    print('')
                iter_num += 1
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data, sample_dir=sample_dir,select_num=stage)  # eval_data value range is 0-255

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
        

    def test(self, test_files, ckpt_dir, save_dir,):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        with tf.variable_scope("", reuse=True):
            temp = tf.get_variable('d1')
            print(self.sess.run((temp)))
            temp = tf.get_variable('denoise_cnn/d20')
            print(self.sess.run((temp)))
            temp = tf.get_variable('denoise_cnn/d30')
            print(self.sess.run((temp)))
        for idx in range(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            output_clean_image, noisy_image = self.sess.run([self.Y_list[-1], self.X],
                                                            feed_dict={self.Y_: clean_image, self.is_training: False, self.size_m:clean_image.shape[1]//8})
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
        
        
