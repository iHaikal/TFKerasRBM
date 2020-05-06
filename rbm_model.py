import tensorflow as tf
from tensorflow.python.keras import layers, Model, backend, losses

def bernoulli_sample(prob):
    return tf.nn.relu(tf.sign(prob - backend.random_uniform(tf.shape(prob))))

class RBM(Model):
    
    def __init__(self,
                 visible_units,
                 hidden_units,
                 name='rbm'):
        super(RBM, self).__init__(name=name)
        
        self.w = self.add_weight(shape=(visible_units, hidden_units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name="w"
                                 )
        self.vb = self.add_weight(shape=(visible_units, ),
                                  initializer='zeros',
                                  trainable=True,
                                  name="vb")
        self.hb = self.add_weight(shape=(hidden_units, ),
                                  initializer='zeros',
                                  trainable=True,
                                  name="hb")
        
        self.mse = losses.MeanSquaredError()
        
        self.v_state = tf.Variable(0.0)
        self.v_prob = tf.Variable(0.0)
        self.h0_prob = tf.Variable(0.0)
        self.h0_state = tf.Variable(0.0)
        self.h1_prob = tf.Variable(0.0)
        self.h1_state = tf.Variable(0.0)
    
    def dense(self, state, w, b):
        prob = tf.sigmoid(tf.matmul(state, w) + b)
        new_state = bernoulli_sample(prob)
        
        return prob, new_state
    
    def call(self, inputs):
        self.h0_prob, self.h0_state = self.dense(inputs, self.w, self.hb)
        self.v_prob, self.v_state = self.dense(self.h0_state, tf.transpose(self.w), self.vb)
        
        self.h1_prob, self.h1_state = self.dense(self.v_state, self.w, self.hb)
        
        return self.v_state
    
    def evaluate(self, inputs):
        return self.mse(inputs, self.call(inputs))
    
    def train(self, inputs, learning_rate):
        current_loss = self.evaluate(inputs)
        
        dW = tf.matmul(tf.transpose(inputs), self.h0_prob) - \
            tf.matmul(tf.transpose(self.v_state), self.h1_prob)
           
        self.w.assign_add(learning_rate * dW)
        self.vb.assign_add(learning_rate * tf.reduce_mean(inputs - self.v_state, 0))
        self.hb.assign_add(learning_rate * tf.reduce_mean(self.h0_state - self.h1_state, 0)) 
        
        return current_loss
        
        
        