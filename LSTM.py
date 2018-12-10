import tensorflow as tf
class Basic_Model(object):
    def __init__(self,hidden_state,input_dim):
        self.batch_view_ph = tf.placeholder(tf.float32, [None, None], name='mid_his_batch_ph')
        self.batch_feature_ph = tf.placeholder(tf.float32, [None, None], name='mid_topo_batch_ph')

        self.hidden_state = hidden_state
        self.input_dim = input_dim
        self.lr = tf.placeholder(tf.float64, [])
    def save_embedding_weight(self, sess):
        embedding = sess.run(self.mid_embeddings_var)
        return embedding
    def train_and_loss(self,inps):
        self.loss = tf.nn.l2_loss(inps-self.batch_feature_ph)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    def train(self, sess, inps):
        if self.use_negsampling:
            loss, _ = sess.run([self.loss,self.optimizer],
                                                       feed_dict={
                                                           self.batch_view_ph: inps[0],
                                                           self.batch_feature_ph: inps[1],
                                                           self.lr: inps[2],
                                                       })
        return loss
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


    class GRUCell(tf.nn.rnn_cell.RNNCell):
        """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

        def __init__(self, num_units, num_input):
            self._num_units = num_units
            self._num_input = num_input

        @property
        def state_size(self):
            return self._num_units

        @property
        def output_size(self):
            return self._num_units

        def __call__(self, inputs, state, scope=None):
            with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"


                with tf.variable_scope("Gates"):  # Reset gate and update gate.
                    # We start with bias of 1.0 to not reset and not update.
                    self.U_u = tf.get_variable("view_hidden_u", [self._num_input, self._num_units])
                    self.W_u = tf.get_variable("hidden_to_hidden_u", [self._num_units, self._num_units])
                    self.b_u = tf.get_variable("bais_u", [self._num_units])
                    u = tf.matmul(inputs, self.W_u) + tf.matmul(state, self.U_u) + self.b_u
                    u = tf.nn.sigmoid(u)
                    self.U_r = tf.get_variable("view_hidden_r", [self._num_input, self._num_units])
                    self.W_r = tf.get_variable("hidden_to_hidden_r", [self._num_units, self._num_units])
                    self.b_u = tf.get_variable("bais_r", [self._num_units])
                    r = tf.matmul(inputs, self.W_r) + tf.matmul(state, self.U_r) + self.b_r
                    r = tf.nn.sigmoid(r)
                with tf.variable_scope("Main"):
                    self.U = tf.get_variable("view_hidden", [self._num_input, self._num_units])
                    self.W = tf.get_variable("hidden_to_hidden", [self._num_units, self._num_units])
                    b = tf.get_variable("bais", [self._num_units])
                    c = tf.nn.relu(tf.matmul(inputs, self.U)+tf.matmul(r * state,self.W)+b)
                new_h = u * state + (1 - u) * c
            return new_h, new_h








class LISTA(Basic_Model):
    def __init__(self,hidden_state,input_dim,hidden_layer):
        super(LISTA, self).__init__(hidden_state,input_dim)

        # (Bi-)RNN layer(-s)

        input_array = tf.reshape(self.batch_view_ph,[-1,1,hidden_state])
        input_array = tf.transpose(input_array,[1,0,2])
        input_array_one = tf.reshape(self.batch_view_ph, [-1, 1, hidden_state])
        input_array_one = tf.transpose(input_array_one, [1, 0, 2])
        for i in range(0,hidden_layer-1):
            input_array.append(self.batch_view_ph)
            input_array = tf.concat([input_array,input_array_one],axis=0)
        with tf.name_scope('rnn_1'):
            basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_state)
            output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, input_array, dtype=tf.float32)
            inp = output_seqs[0,:,:]
        self.train_and_loss(inp)
        # Fully connected layer