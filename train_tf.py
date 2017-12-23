import tensorflow as tf
import model as mod
import input_data

class Train():

    def __init__(self, model, learning_rate,epochs,batch, display_step):

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        #params
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch  = batch
        self.display_step = 1
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

    def get_model(self, model, x, y):

        if (model == 'linear_regression'):
            model,cost_ftn = mod.Model(x = x, y = y).get_linear_model()

        if (model == 'cnn'):
            m_class = mod.Model(x = x, y = y)
            model,cost_ftn = m_class.get_cnn_model()

        return model, cost_ftn

model = Train(model = 'cnn', learning_rate = 0.001,epochs = 5,batch =20, display_step = 1)
model.train()


