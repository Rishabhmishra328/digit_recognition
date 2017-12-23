import tensorflow as tf
import input_data


class Model():

    def __init__(self, model, learning_rate,epochs,batch, display_step):

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        #params
        self.learning_rate = 0.01
        self.epochs = 50
        self.batch  = 100
        self.display_step = 2
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.model, self.cost_function = self.get_model(model, self.x, self.y)

    def train(self):

       #gradient descent
       optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_function)

       #initializing
       init = tf.global_variables_initializer()

       #firing up session
       with tf.Session() as sess:
           sess.run(init)

           #training
           for iteration in range (self.epochs):
               avg_cost = 0.
               total_batch = self.mnist.train.num_examples/self.batch
               #looping over batches
               for i in range(total_batch):
                   batch_x, batch_y = self.mnist.train.next_batch(self.batch)
                   #fitting data
                   sess.run(optimizer, feed_dict = {self.x: batch_x, self.y: batch_y})
                   #calculating total loss
                   avg_cost += sess.run(self.cost_function, feed_dict = {self.x: batch_x, self.y: batch_y})/total_batch
               #display logs each iteration
               if iteration % self.display_step == 0 :
                   print('Iteration : ', '%04d' % (iteration + 1), 'cost = ', '{:.9f}'.format(avg_cost))

           print('Training complete')
    
           #testing
           predictions = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1)) 
           #accuracy
           accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
           print('Accuracy : ', accuracy.eval({self.x: self.mnist.test.images, self.y: self.mnist.test.labels}))

    def get_model(self, model, X, Y):

        # if (model == 'cnn')
        #model
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        model = tf.nn.softmax(tf.matmul(X, W) + b)
    
        #cross entropy
        cost_ftn = -tf.reduce_sum(Y * tf.log(model))

        return model, cost_ftn


