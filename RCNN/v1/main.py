import os
import argparse
from glob import glob

import tensorflow as tf

from model import denoiser
from utils import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=150, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--kstage', dest='kstage', type=int, default=3, help='# number of kstage')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--sigma', dest='sigma', type=int, default=25, help='noise level')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='Set12', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='Set12', help='dataset for testing')
parser.add_argument('--stay_epoch', dest ='stay_epoch',type=int, default=5,help='The epoch number of stay')
parser.add_argument('--dcnn_layer', dest ='dcnn_layer',type=int, default=0,help='dcnn layer flag, 0 for full, 1 for half, 2 for quarter')
parser.add_argument('--finetune', dest ='finetune',type=int,default=0,help='finetune delta flag, 0 for false, 1 for given, 2 for true')
parser.add_argument('--pretrain', dest = 'pretrain',type=int,default=2,help='pretrain net flag, 0 for false, 1 for jump , 2 for step')
parser.add_argument('--kstar', dest = 'kstar',type=int,default=0,help='stage for jump in pretrain')
parser.add_argument('--load', dest = 'load',type=int,default=0,help='load model flag, 1 for not load, 0 for load')
args = parser.parse_args()

def denoiser_train(denoiser, lr_init, lr_decay):
    with load_data(filepath='./data/img_clean_pats.npy') as data:
        # if there is a small memory, please comment this line and uncomment the line99 in model.py
        #data = data.astype(np.float32) / 255.0  # normalize the data to 0-1
        eval_files = glob('./data/test/{}/*.png'.format(args.eval_set))
        eval_data = load_images(eval_files)  # list of array of different size, 4-D, pixel value range is 0-255
        print '..........................................'
        denoiser.train(data, eval_data, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr_init=lr_init, lr_decay = lr_decay
                       ,config = args,sample_dir=args.sample_dir)

def denoiser_test(denoiser):
    test_files = glob('./data/test/{}/*.png'.format(args.test_set))
    denoiser.test(test_files, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    print(args)
    lr_init = args.lr
    lr_decay = [1.0,0.2]
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(allow_growth=True)
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        print(gpu_options)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess, sigma=args.sigma,kstage = args.kstage,batch_size=args.batch_size,config = args)
            if args.phase == 'train':
                denoiser_train(model, lr_init=lr_init, lr_decay = lr_decay)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, sigma=args.sigma)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()
