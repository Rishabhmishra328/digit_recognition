import tensorflow as tf

class Model():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_linear_model(self):
        #model
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        model = tf.nn.softmax(tf.matmul(self.x, W) + b)
    
        #cross entropy
        cost_ftn = -tf.reduce_sum(self.y * tf.log(model))

        return model, cost_ftn

    def get_cnn_model(self):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.reshape(self.x, [-1, 28, 28, 1])
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], kernel_initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32) , padding="same", activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], kernel_initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32), padding="same", activation=tf.nn.relu)
  
        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        # Calculate Loss (for both TRAIN and EVAL modes)

        # onehot_labels = tf.one_hot(indices=tf.cast(self.y, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels= self.y, logits=logits)

        return logits, loss

