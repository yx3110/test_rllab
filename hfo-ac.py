#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import json

import tensorflow as tf
from keras import backend as K

from ActorNet import ActorNet
from CriticNet import CriticNet
from Utils import *

np.random.seed(42)

batch_size = 32  # batch size for training
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
evaluate_e = 0  # Epsilon used in evaluation
discount_factor = 0.99
annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 30000  # How many episodes of game environment to train network with.
pre_train_steps = 1000  # How many steps of random actions before training begins.
num_players = 1
num_opponents = 0
tau = 0.001  # Tau value used in target network update
num_features = (58 + (num_players - 1) * 8 + num_opponents * 8) * num_players
step_counter = 0
load_model = False  # Load the model
use_gpu = True
train = True
if train:
    e = startE
else:
    e = evaluate_e
step_drop = (startE - endE) / annealing_steps

if use_gpu:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
else:
    config = tf.ConfigProto()
exp_buffer = ExpBuffer()
total_reward = 0
sess = tf.Session(config=config)

K.set_session(sess)
reward_buffer = []
reward_buffer_size = 100

actor = ActorNet(team_size=num_players, enemy_size=num_opponents, tau=tau, sess=sess)
critic = CriticNet(team_size=num_players, enemy_size=num_opponents, tau=tau, sess=sess)

# init model by creating new model or loading
# activate HFO agent(s)
# players = []
print("Loading the weights")
if load_model:
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

env = hfoENV()

for episode in range(num_episodes):
    while env.game_info.status == IN_GAME:
        loss = 0
        # Grab the state features from the environment
        state0 = env.env.getState()
        action_arr = actor.model.predict(np.reshape(state0, [1, num_features]))[0]
        dice = np.random.uniform(0, 1)
        if dice < e and train:
            print "Random action is taken for exploration, e = " + str(e)
            new_action_arr = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                              np.random.uniform(0, 100), np.random.uniform(-180, 180), np.random.uniform(-180, 180),
                              np.random.uniform(0, 100), np.random.uniform(-180, 180)]
            action_arr = new_action_arr
        if train and e >= endE and exp_buffer.cur_size >= pre_train_steps:
            e -= step_drop

        # Take an action and get the current game status
        state1, reward, done, _ = env.step(action_arr)
        print action_arr

        # Fill buffer with record
        exp_buffer.add(
            Experience(state0=state0, action=action_arr, state1=state1, reward=reward, done=done))
        # Train the network
        if exp_buffer.cur_size >= pre_train_steps and train:
            # sample batch
            cur_experience_batch = exp_buffer.sample(batch_size)
            state0s = np.asarray([cur_exp.state0 for cur_exp in cur_experience_batch])
            actions = np.reshape(np.asarray([cur_exp.action for cur_exp in cur_experience_batch]), [batch_size, 8])
            rewards = np.asarray([cur_exp.reward for cur_exp in cur_experience_batch])
            dones = np.asarray([cur_exp.done for cur_exp in cur_experience_batch])
            state1s = np.asarray([cur_exp.state1 for cur_exp in cur_experience_batch])
            ys = np.zeros((state0s.shape[0], 1))

            target_q_values = critic.target_model.predict([state1s, actor.target_model.predict(state1s)])

            for k in xrange(batch_size):
                if dones[k]:
                    ys[k] = rewards[k]
                else:
                    ys[k] = rewards[k] + discount_factor * target_q_values[k]
            if train:
                loss += critic.model.train_on_batch([state0s, actions], ys)
                actions_for_grad = actor.model.predict(state0s)
                grads = critic.gradients(state0s, actions_for_grad)
                actor.update(state0s, grads)
                actor.target_train()
                critic.target_train()

        step_counter += 1

        print("Episode", episode + 1, "Step", step_counter, "Reward", reward, "Loss", loss)

    # Check the outcome of the episode
    total_reward += env.game_info.total_reward
    print('Episode %d ended with %s' % (episode + 1, env.env.statusToString(env.game_info.status)))
    print("Episodic TOTAL REWARD @ " + str(episode + 1) + "-th Episode  : " + str(env.game_info.total_reward))
    reward_buffer.append(env.game_info.total_reward)
    if len(reward_buffer) > reward_buffer_size:
        reward_buffer.pop(0)
    total_over_last100 = 0
    for r in reward_buffer:
        total_over_last100 += r

    average = np.divide(total_over_last100, len(reward_buffer))

    print("Total REWARD: ", total_reward, "EOT Reward", env.game_info.extrinsic_reward,
          "average reward over last 100 episodes:", average)
    if np.mod(episode, 100) == 0 and train:
        actor.model.save_weights('actormodel' + str(episode) + '.h5', overwrite=True)
        with open("actormodel.json", "w") as outfile:
            json.dump(actor.model.to_json(), outfile)

        critic.model.save_weights('criticmodel' + str(episode) + '.h5', overwrite=True)
        with open("criticmodel.json", "w") as outfile:
            json.dump(critic.model.to_json(), outfile)
    env.game_info.reset()
    # Quit if the server goes down

    if done == SERVER_DOWN:
        env.act(QUIT)
        break
    episode += 1
