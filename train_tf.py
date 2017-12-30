import tensorflow as tf
import model as mod
import input_data
import cv2
import numpy as np

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


    def predict_segment(self, filename , imgpixel):
      filepath = filename + '.jpg'
      img = cv2.imread(filepath)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      img = cv2.bitwise_not(img)
      img = img.astype(dtype = np.float32)/255
      reduction_factor = 28.0 / imgpixel
      img  = cv2.resize(img, ((int)(img.shape[0] * (reduction_factor)), (int)(img.shape[1] * (reduction_factor))))
      # Input Layer
      input_layer = tf.reshape(img, [-1, img.shape[0], img.shape[1], 1])
      # Convolutional Layer #1
      conv = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[28, 28], kernel_initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32) , padding="same", activation=tf.nn.relu)
      print(tf.cast(input_layer, dtype = tf.float32))
      print(conv.shape)
      return img

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
                   # print(batch_x[0].shape)
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
           digit = self.predict_segment('digit_color', 100)

    def get_model(self, model, x, y):

        if (model == 'linear_regression'):
            model,cost_ftn = mod.Model(x = x, y = y).get_linear_model()

        if (model == 'cnn'):
            m_class = mod.Model(x = x, y = y)
            model,cost_ftn = m_class.get_cnn_model()

        return model, cost_ftn

    def predict_digit(self, img):
        print(img.shape)
        #testing
        predictions = tf.argmax(self.model, 1)
        with tf.Session() as sess:

           sess.run(init)

           prediction = sess.run(self.model, feed_dict = {self.x: img})
           print('Confidence : ', prediction)


model = Train(model = 'linear_regression', learning_rate = 0.01,epochs = 50,batch =20, display_step = 1)
model.train()


