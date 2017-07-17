import keras.backend as K
import tensorflow as tf
from keras import initializers
from keras import layers
from keras.engine import Model
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Nadam

max_turn_angle = 180
min_turn_angle = -180
max_power = 100
min_power = 0


class ActorNet:
    def __init__(self, sess, tau, learning_rate=0.00001, team_size=1, enemy_size=0, ):
        self.sess = sess
        self.TAU = tau
        K.set_session(sess)
        self.learning_rate = learning_rate

        self.input_size = (58 + (team_size - 1) * 8 + enemy_size * 8) * team_size
        self.relu_neg_slope = 0.01
        self.model, self.weights, self.state = self.create_actor_network(self.input_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(self.input_size)

        self.action_gradient = tf.placeholder(tf.float32, [None, 8])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def update(self, state, grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: state,
            self.action_gradient: grads
        })

    def create_actor_network(self, state_size):
        print("Building actor model")
        actor_input = Input(shape=[state_size], name='actor_in')
        dense1 = Dense(512, kernel_initializer=initializers.glorot_normal(),
                       bias_initializer='zeros', name='actor_d1')(
            actor_input)
        relu1 = LeakyReLU(alpha=self.relu_neg_slope, name='actor_re1')(dense1)
        dense2 = Dense(256, kernel_initializer=initializers.glorot_normal(),
                       bias_initializer='zeros', name='actor_d2')(
            relu1)
        relu2 = LeakyReLU(alpha=self.relu_neg_slope, name='actor_re2')(dense2)
        dense3 = Dense(128, kernel_initializer=initializers.glorot_normal(),
                       bias_initializer='zeros', name='actor_d3')(
            relu2)
        relu3 = LeakyReLU(alpha=self.relu_neg_slope, name='actor_re3')(dense3)
        dense4 = Dense(64, kernel_initializer=initializers.glorot_normal(),
                       bias_initializer='zeros', name='actor_d4')(
            relu3)
        relu4 = LeakyReLU(alpha=self.relu_neg_slope, name='actor_re4')(dense4)
        action_out = Dense(3, kernel_initializer=initializers.glorot_normal(),
                           bias_initializer='zeros', name='actor_aout')(
            relu4)
        param_out = Dense(5, kernel_initializer=initializers.glorot_normal(),
                          bias_initializer='zeros', name='actor_pout')(
            relu4)
        actor_out = layers.concatenate([action_out, param_out], axis=1)
        print 'here3'
        model = Model(inputs=actor_input, outputs=actor_out)
        adam = Nadam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return model, model.trainable_weights, actor_input
