import sys
from LSTM import *
from sparse_data_generate import *
sparse_vec_dim = 100
dense_vec_dim = 50
test_rate = 0.1
sparse_rate = 0.1
SNR = 30
maxsample = 50000
batch_size = 256
data_dir = sys.path[0]

Sparse_path = generate_sparse_data(dense_vec_dim,sparse_vec_dim,test_rate,data_dir,dict_type = 'rand',nonnegtive = True,sparse_rate = 0.1,noise_type = 'guass',noise_level = SNR,sparse_type = 'uniform')
print Sparse_path
#Sparse_path = 'sparse_' + str(sparse_vec_dim) + '_' + str(dense_vec_dim) + '_'+str(SNR)+'_'+str(int(sparse_rate))+'_'+'data.h5'


regression = True
train_set, valid_set, test_set = load_Sparse_data(Sparse_path, valid_portion=0.1, maxlen=maxsample,regression=regression)
print type(train_set)
a,b=train_set
print b[0]
GRU = LISTA(hidden_state = sparse_vec_dim,input_dim = dense_vec_dim,hidden_layer= 10)


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    train_shuffle = get_minibatches_idx(len(train_set[0]), batch_size, shuffle=True)
    for _, train_index in train_shuffle:
        x = [train_set[0][t] for t in train_index]
        y = [train_set[1][t] for t in train_index]
        x, y = prepare_data(x, y)
        loss = GRU.train(sess=sess)
        print loss