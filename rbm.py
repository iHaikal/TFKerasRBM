import tensorflow as tf
from tensorflow.python.keras import layers, Model, backend, losses
from enum import Enum

def bernoulli_sample(prob):
    return tf.nn.relu(tf.sign(prob - backend.random_uniform(tf.shape(prob))))

class DenseRBM(layers.Layer):
    
    def __init__(self,
                 units,
                 name='rbm_dense_layer'):
        super(DenseRBM, self).__init__(name=name)
        
        self.b = self.add_weight(shape=(units, ),
                                 initializer='zeros',
                                 trainable=True,
                                 name="b")
    
    def prob_dist(self, inputs, weights):
        return tf.sigmoid(tf.matmul(inputs, weights) + self.b)
    
    def call(self, inputs, weights):
        prob = self.prob_dist(inputs, weights)
        state = bernoulli_sample(prob)
        
        return prob, state

class PredictedOutput(Enum):
    both = 'all'
    hidden = 'hidden'
    visible = 'visible'

class RBM(Model):
    
    i = 0 # fliping index for computing pseudo_likelihood
    
    def __init__(self,
                 visible_units,
                 hidden_units,
                 momentum=0.95,
                 k=1,
                 name='rbm'):
        super(RBM, self).__init__(name=name)
        
        self.w = self.add_weight(shape=(visible_units, hidden_units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name="w"
                                 )
        
        self.visible_units = visible_units
        
        self.v = DenseRBM(visible_units)
        self.h = DenseRBM(hidden_units)
        self.k = k

        self.momentum = momentum
        
        self.dW = tf.Variable(tf.zeros((visible_units, hidden_units)), dtype=tf.float32)
        self.dVB = tf.Variable(tf.zeros((visible_units, )), dtype=tf.float32)
        self.dHB = tf.Variable(tf.zeros((hidden_units, )), dtype=tf.float32)
        
        self.mse = losses.MeanSquaredError()
    
    def call(self, inputs):
        h_prob, h_state = self.h(inputs, weights=self.w)
        v_prob, v_state = self.v(h_state, weights=tf.transpose(self.w))
        
        return (h_prob, h_state) , (v_prob, v_state)
    
    def predict(self, inputs, predict_output=PredictedOutput.both): 
        h_prob, h_state, v_prob, v_state = super(RBM, self).predict(inputs)
        
        if predict_output == PredictedOutput.both:
            return (h_prob, h_state), (v_prob, v_state) 
        elif predict_output == PredictedOutput.hidden:
            return (h_prob, h_state)
        else:
            return (v_prob, v_state)
        
    
    def CD_k(self, inputs):
        v = inputs
        h0, h = self.h(v, weights=self.w)
        
        trans_w = tf.transpose(self.w)
        for _ in range(self.k):
            _, v = self.v(h, weights=trans_w)
            _, h = self.h(v, weights=self.w)
        
        dW = tf.matmul(tf.transpose(inputs), h0) - tf.matmul(tf.transpose(v), h)
        dHB = tf.reduce_mean(h0 - h, 0)
        dVB = tf.reduce_mean(inputs - v, 0)
        
        return dW, dHB, dVB
    
    def free_energy(self, inputs):
        wx_b = tf.matmul(inputs, self.w) + self.h.b
        vbias_term = tf.matmul(inputs, tf.reshape(self.v.b, [tf.shape(self.v.b)[0], 1]))
        hidden_term = tf.reduce_sum(tf.math.log(1 + tf.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term 
    
    def pseudo_likelihood(self, inputs):
        x = tf.round(inputs)
        x_fe = self.free_energy(x)
        
        split0, split1, split2 = tf.split(x, [self.i, 1, tf.shape(x)[1] - self.i - 1], 1)
        xi = tf.concat([split0, 1 - split1, split2], 1)
        self.i = (self.i + 1) % self.visible_units
        xi_fe = self.free_energy(xi)
        
        return tf.reduce_mean(self.visible_units * tf.math.log(tf.sigmoid(xi_fe - x_fe)), axis=0)
    
    def evaluate(self, inputs):
        return self.pseudo_likelihood(inputs)
    
    def new_delta(self, old, new):
        return old * self.momentum + new * (1 - self.momentum)
    
    def train(self, inputs, learning_rate):
        current_loss = self.evaluate(inputs)
        
        dW, dHB, dVB = self.CD_k(inputs)
        
        self.dW = self.new_delta(self.dW, learning_rate *  dW)
        self.dVB = self.new_delta(self.dVB, learning_rate * dVB)
        self.dHB = self.new_delta(self.dHB, learning_rate * dHB)
        
        self.w.assign_add(self.dW)
        self.v.b.assign_add(self.dVB)
        self.h.b.assign_add(self.dHB) 
        
        return current_loss
        
        
        