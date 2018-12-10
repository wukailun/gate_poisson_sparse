import sys

import numpy as np
import os
import h5py


def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower) + x
def generor_data(dense_vec_dim, sparse_vec_dim,dictionary ,max_sample,nonnegtive = True, sparse_type = 'uniform' ,SNR = 30, sparse_rate = 0.1,noise_type = 'guass'):


    'sparse_' + str(sparse_vec_dim) + '_' + str(dense_vec_dim) + '_' + str(SNR) + '_' + str(int(sparse_rate * sparse_vec_dim)) + '_' + 'data.h5'
    np.zeros(shape=(max_sample,dense_vec_dim))
    y_benchmark_out = np.zeros(shape=(max_sample,dense_vec_dim))
    y_out = np.zeros(shape=(max_sample,dense_vec_dim))
    x_orign_out = np.zeros(shape=(max_sample,sparse_vec_dim))
    for i in range(0,max_sample):
        if sparse_type == 'uniform':
            x_orign = np.random.uniform(low=0, high=1.0, size=sparse_vec_dim)
            if not nonnegtive:
                x_orign = x_orign - 0.5
        elif sparse_type == 'guass':
            x_orign = np.random.normal(loc = 0, scale = 1, size = sparse_vec_dim)
            if nonnegtive:
                x_orign = np.abs(x_orign)
        x = np.random.uniform(low=0, high=1.0, size=sparse_vec_dim)
        x_index = np.where(x > sparse_rate)
        x_orign[x_index] = 0
        y_benchmark = np.dot(dictionary,x_orign)
        y = wgn(y_benchmark,SNR)
        y_benchmark_out[i,:] = y_benchmark
        y_out[i,:] = y
        x_orign_out[i,:] = x_orign
    return x_orign_out,y_out,y_benchmark_out

def save_file(x, y, data_dir, dictionary, test_rate, SNR, sparse_rate):
    data_fn = data_dir +'/'+ 'sparse_'+str(x.shape[1])+'_'+str(y.shape[1])+'_'+str(SNR)+'_'+str(int(sparse_rate*x.shape[1]))+ '_data.h5'
    if os.path.exists(data_fn):
        print 'the data file exists'
        return data_fn
    if os.path.exists(data_fn):
        os.remove(data_fn)
    print data_fn
    data_file = h5py.File(data_fn)
    dim1_x,dim2_x = x.shape
    train_dim = int(dim1_x*(1-test_rate))
    data_file['train_origin'] = x[:train_dim,:]
    data_file['train_view'] = y[:train_dim,:]
    data_file['test_origin'] = x[train_dim:,:]
    data_file['test_view'] = y[train_dim:,:]
    data_file['A'] = dictionary
    data_file.close()
    return
def dictionary_gener(dense_vec_dim,sparse_vec_dim,dict_type = 'rand'):
    if dict_type == 'rand':
        A = np.random.normal(loc = 0, scale = 1.0, size = [dense_vec_dim,sparse_vec_dim])/np.sqrt(dense_vec_dim)
    elif dict_type == 'Fourier':
        A = np.fft.fft(np.eye(sparse_vec_dim))
    elif dict_type == 'Gabor':
        A = 0
    return A
def generate_sparse_data(dense_vec_dim,sparse_vec_dim,test_rate,data_dir,max_sample = 60000,dict_type = 'rand'\
                         ,nonnegtive = True,sparse_rate = 0.1,noise_type = 'guass',noise_level = pow(10,-3),sparse_type = 'uniform'):
    dict = dictionary_gener(dense_vec_dim, sparse_vec_dim, dict_type = dict_type)
    x,y,y_benchmark = generor_data(dense_vec_dim,sparse_vec_dim,dict,max_sample = max_sample,nonnegtive = nonnegtive ,sparse_type = sparse_type, SNR = noise_level, sparse_rate = sparse_rate, noise_type = noise_type)
    return save_file(x, y, data_dir, dict, test_rate,noise_level,sparse_rate)





def load_Sparse_data(Sparse_path , valid_portion=0.1, maxlen=None,regression = False):
    Sparse_File = h5py.File(Sparse_path)
    train_y_candidate = np.array(Sparse_File['train_origin'])
    train_x = np.array(Sparse_File['train_view'])
    test_y_candidate= np.array(Sparse_File['test_origin'])
    test_x = np.array(Sparse_File['test_view'])
    #if it is not a regression model, convert the value large than 0 to 1
    if not regression:
        train_y_candidate[np.where(train_y_candidate > 0)] = 1
        test_y_candidate[np.where(test_y_candidate > 0)] = 1
    [n, dimen] = train_y_candidate.shape
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for i in range(0,n):
            if i < maxlen:
                x = train_x[i,:].tolist()
                y = train_y_candidate[i,:].tolist()
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y
    else:
        new_train_set_x = []
        new_train_set_y = []
        for i in range(0,n):
            x = train_x[i, :].tolist()
            y = train_y_candidate[i, :].tolist()
            new_train_set_x.append(x)
            new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y
    """ Step 3: split train set into valid set according to the 'valid_portion'
    """
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)

    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    """ Step 4: remove words that exceed the most
                frequent indexes given by 'n_words'
    """
    new_test_set_x = []
    new_test_set_y = []
    [n, dimen] = test_y_candidate.shape
    for i in range(0, n):
        x = test_x[i, :].tolist()
        y = test_y_candidate[i, :].tolist()
        new_test_set_x.append(x)
        new_test_set_y.append(y)
        test_set = [new_test_set_x,new_test_set_y]
    return train_set, valid_set, test_set


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    This function shuffles the samples at the
    beginning of each iteration

    Parameters
    ----------
    :type n: int
    :param n: number of samples

    :type minibatch_size: int
    :param minibatch_size:

    :type shuffle: bool
    :param shuffle: whether to shuffle the samples
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    # zipped (index, list) pair
    return zip(range(len(minibatches)), minibatches)
##

def prepare_data(seqs, labels=None,regression = True):
    """ Create the matrices from datasets. specially for those have labels

        This pad each sequence to the same length: the
        length of the longest sequence or maxlen

        :param seqs: list of list, lists of sentences,
                     each sentence with different length
        :param labels: list of labels

        :return:
        :type x: ndarray with size (maxlen, n_samples)
        :param x: data fed into the rnn

        :type x_mask: ndarray with size (maxlen, n_samples)
        :param x_mask: mask for the data matrix 'x'

        :param labels: list of lables just as the input
    """
    seqs = list(seqs)
    lengths = [len(s) for s in seqs]




    n_samples = len(seqs)


    x_deminsion = len(seqs[0])
    y_deminsion = len(labels[0])
    # each column in x corresponding to a sample
    x = np.zeros((n_samples, x_deminsion)).astype(theano.config.floatX)
    if regression:
        labels_out = np.zeros((n_samples, y_deminsion)).astype(theano.config.floatX)
    else:
        labels_out = np.zeros((n_samples, y_deminsion)).astype(np.int64)
    for idx, s in enumerate(seqs):
        x[idx, :] = s
    if labels != None:
        for idx, s in enumerate(labels):

            labels_out[idx, :] = s

    return x , labels_out