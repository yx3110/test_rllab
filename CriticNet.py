import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import layers
from keras.engine import Model
from keras.layers import Input
from keras.layers import LeakyReLU, initializers
from keras.layers.core import Dense
from keras.optimizers import Nadam

max_turn_angle = 180
min_turn_angle = -180
max_power = 100
min_power = 0
max_act_val = 1
min_act_val = -1


def bound(grad, param, max_val, min_val):
    if grad > 0:
        return grad * np.divide(max_val - param, max_val - min_val)
    else:
        return grad * np.divide(param - min_val, max_val - min_val)


def bound_grads(cur_grads, cur_actions, index):
    if 0 <= index < 3:
        cur_grads[index] = bound(cur_grads[index], cur_actions[index], max_act_val, min_act_val)
    elif index == 3 or index == 6:
        cur_grads[index] = bound(cur_grads[index], cur_actions[index], max_power, min_power)
    elif index == 4 or index == 5 or index == 7:
        cur_grads[index] = bound(cur_grads[index], cur_actions[index], max_turn_angle, min_turn_angle)


class CriticNet:
    def __init__(self, sess, tau, learning_rate=0.001, team_size=1, enemy_size=0):
        self.TAU = tau
        self.relu_neg_slope = 0.01
        self.learning_rate = learning_rate
        self.input_size = (58 + (team_size - 1) * 8 + enemy_size * 8) * team_size
        self.sess = sess
        K.set_session(sess)

        self.model, self.action, self.state = self.create_critic_network(self.input_size, 8)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(self.input_size, 8)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        grads = self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]
        #  print "grads:"+str(grads)+"len of grads:"+str(len(grads))+"len of actions:"+str(len(actions))
        for i in xrange(len(grads)):
            for j in xrange(len(grads[i])):
                bound_grads(grads[i], actions[i], j)
        return grads

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        print("Building critic model")
        critic_input_action = Input(shape=[action_dim], name='critic_ain')
        critic_input_state = Input(shape=[state_size], name='critic_sin')
        critic_input_final = layers.concatenate([critic_input_state, critic_input_action], axis=1, name='critic_in')
        dense1 = Dense(1024, kernel_initializer=initializers.glorot_normal(),
                       bias_initializer='zeros', name='critic_d1')(critic_input_final)
        relu1 = LeakyReLU(alpha=self.relu_neg_slope, name='critic_re1')(dense1)
        dense2 = Dense(512, kernel_initializer=initializers.glorot_normal(),
                       bias_initializer='zeros', name='critic_d2')(relu1)
        relu2 = LeakyReLU(alpha=self.relu_neg_slope, name='critic_re2')(dense2)
        dense3 = Dense(256, kernel_initializer=initializers.glorot_normal(),
                       bias_initializer='zeros', name='critic_d3')(relu2)
        relu3 = LeakyReLU(alpha=self.relu_neg_slope, name='critic_re3')(dense3)
        dense4 = Dense(128, kernel_initializer=initializers.glorot_normal(),
                       bias_initializer='zeros', name='critic_d4')(relu3)
        relu4 = LeakyReLU(alpha=self.relu_neg_slope, name='critic_re4')(dense4)
        critic_out = Dense(1, kernel_initializer=initializers.glorot_normal(),
                           bias_initializer='zeros')(relu4)

        model = Model(inputs=[critic_input_state, critic_input_action], outputs=critic_out)
        adam = Nadam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return model, critic_input_action, critic_input_state
