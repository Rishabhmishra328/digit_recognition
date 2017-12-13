import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#params
learning_rate = 0.01
epochs = 50
batch  = 100
display_step = 2


def train():
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    #model
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    

    with tf.name_scope('Wx_b') as scope:
        #linear model
        model = tf.nn.softmax(tf.matmul(x, W) + b)


    with tf.name_scope('cost_function') as scope:
        #cross entropy
        cost_function = -tf.reduce_sum(y * tf.log(model))

    with tf.name_scope('train') as scope:
        #gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    #initializing
    init = tf.global_variables_initializer()

    #firing up session
    with tf.Session() as sess:
        sess.run(init)

        #training
        for iteration in range (epochs):
            avg_cost = 0.
            total_batch = mnist.train.num_examples/batch
            #looping over batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch)
                #fitting data
                sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
                #calculating total loss
                avg_cost += sess.run(cost_function, feed_dict = {x: batch_x, y: batch_y})/total_batch
            #display logs each iteration
            if iteration % display_step == 0 :
                print('Iteration : ', '%04d' % (iteration + 1), 'cost = ', '{:.9f}'.format(avg_cost))

        print('Training complete')

        #testing
        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1)) 
        #accuracy
        accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
        print('Accuracy : ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train()


