import tensorflow as tf
from tensorflow.python.keras import layers, Model, backend, losses

def bernoulli_sample(prob):
    return tf.nn.relu(tf.sign(prob - backend.random_uniform(tf.shape(prob))))

class RBM(Model):
    
    def __init__(self,
                 visible_units,
                 hidden_units,
                 momentum=0.95,
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
        
        self.momentum = momentum
        
        self.dW = tf.Variable(tf.zeros((visible_units, hidden_units)), dtype=tf.float32)
        self.dVB = tf.Variable(tf.zeros((visible_units, )), dtype=tf.float32)
        self.dHB = tf.Variable(tf.zeros((hidden_units, )), dtype=tf.float32)
        
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
    
    def new_delta(self, old, new):
        return old * self.momentum + new * (1 - self.momentum)
    
    def train(self, inputs, learning_rate):
        current_loss = self.evaluate(inputs)
        
        dW = learning_rate * (tf.matmul(tf.transpose(inputs), self.h0_prob) - \
            tf.matmul(tf.transpose(self.v_state), self.h1_prob))
        dVB = learning_rate * tf.reduce_mean(inputs - self.v_state, 0)
        dHB = learning_rate * tf.reduce_mean(self.h0_state - self.h1_state, 0)
        
        self.dW = self.new_delta(self.dW, dW)
        self.dVB = self.new_delta(self.dVB, dVB)
        self.dHB = self.new_delta(self.dHB, dHB)
        
        self.w.assign_add(self.dW)
        self.vb.assign_add(self.dVB)
        self.hb.assign_add(self.dHB) 
        
        return current_loss
        
        
        