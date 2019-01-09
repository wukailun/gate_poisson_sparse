import sys
from LSTM import *
from sparse_data_generate import *
sparse_vec_dim = 100
dense_vec_dim = 50
test_rate = 0.1
sparse_rate = 0.1
SNR = 50
maxsample = 500000
batch_size = 256
data_dir = sys.path[0]

Sparse_path,dict = generate_sparse_data(dense_vec_dim,sparse_vec_dim,test_rate,data_dir,dict_type = 'rand',nonnegtive = True,sparse_rate = 0.1,noise_type = 'guass',noise_level = SNR,sparse_type = 'uniform')
print Sparse_path
print dict
#Sparse_path = 'sparse_' + str(sparse_vec_dim) + '_' + str(dense_vec_dim) + '_'+str(SNR)+'_'+str(int(sparse_rate))+'_'+'data.h5'



Sparse_File = h5py.File(Sparse_path)
train_y_candidate = np.array(Sparse_File['train_origin'])
train_x = np.array(Sparse_File['train_view'])
test_y_candidate= np.array(Sparse_File['test_origin'])
test_x = np.array(Sparse_File['test_view'])
dict = np.array(Sparse_File['A'])
print np.sum((np.dot(test_y_candidate,np.transpose(dict))-test_x)**2)






regression = True
train_set, valid_set, test_set = load_Sparse_data(Sparse_path, valid_portion=0.1, maxlen=maxsample,regression=regression)
print type(train_set)
a,b=train_set
print np.sum((np.dot(np.array(valid_set[1]),np.transpose(dict))-np.array(valid_set[0]))**2)
GRU = LISTA(hidden_state = sparse_vec_dim,input_dim = dense_vec_dim,hidden_layer= 20)

min_loss = 10
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    maxepoch = 300
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(0,maxepoch):
        train_shuffle = get_minibatches_idx(len(train_set[0]), batch_size, shuffle=True)
        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())
        loss = 0
        cnt = 0
        for _, train_index in train_shuffle:
            x = [train_set[0][t] for t in train_index]
            y = [train_set[1][t] for t in train_index]
            x, y = prepare_data(x, y)
            x.shape
            cnt+=1
            #print y.shape
            #a,out = GRU.do_test(sess=sess,inps=x)
            #print np.sum((np.dot(np.array(train_set[1]),np.transpose(dict))-np.array(train_set[0]))**2)
	    
	    #print '******' 
	    #print a[0]
	    #print '.................................'
            #print y
            #print '1'
            #print y[0]
            loss += GRU.train(sess=sess,inps = [x, y, 0.001])
        #print a[0][0]
	#a_i = np.where(y[0]>0.00001)
	#print a[0][0][a_i]
	
	#b_i = np.where(out[0]>0.00001)
	#print a[0][0][b_i]
	test_loss = 0
	test_cnt = 0
	
	valid_shuffle = get_minibatches_idx(len(valid_set[0]), batch_size, shuffle=True)
	for _, train_index in valid_shuffle:
            x = [valid_set[0][t] for t in train_index]
            y = [valid_set[1][t] for t in train_index]
            x, y = prepare_data(x, y)
            x.shape
            test_cnt+=1
            test_loss += GRU.test(sess=sess,inps=[x,y])
	print loss/cnt, test_loss/test_cnt
	if min_loss > test_loss/test_cnt:
	    min_loss = test_loss/test_cnt
	print 'min loss ',min_loss
