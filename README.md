# Restricted Boltzmann Machine implementation in Tensorflow 2.x with Keras

Restricted Boltzmann Machine (shorthanded to RBM) is a type of an Autoencoder. It learns to reconstruct the inputs by themselves in an unsupervised fashion. Unlike traditional autoencoders, RBMs have a stochastic approach towards backpropagation. The implementation is inspired from [[1]](http://deeplearning.net/tutorial/rbm.html)

![Autoencoder vs RBM](https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2018/11/Picture1-4-528x264-528x264.png)


## Training
RBM learns a probability of distribution over the input, so that after trained, it can generate new samples from the learned probability distribution. In order to generate samples similar to the input, it needs to maximise the product of (input) probabilities. This cannot be done using gradient descent, thus it requires an unconventional way of optimization for training. It uses a combination of Gibbs Sampling and Contrastive Divergence to update the weights accordingly [[2]](http://www.cs.utoronto.ca/~hinton/absps/netflixICML.pdf)