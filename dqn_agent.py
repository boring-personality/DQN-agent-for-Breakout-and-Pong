import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model
from collections import deque
import numpy as np
import os
import sys

from helper_functions import *

ENVIRONMENT = 'MountainCar-v0'
REPLAY_MEMORY = 80000			# SIZE OF REPLAY MEMORY EACH ELEMENT WILL CONTAIN (STATE, ACTION, REWARD, NEXT_STATE)
UPDATE_FREQ = 4					# NETWORK WEIGHTS WILL BE UPDATED AT THIS FREQUENCY
UPDATE_TARGET = 1000			# COPY NETWORK WEIGHTS TO TARGET NETWORK
EPSILON_MAX = 1.0				# MAXIMUM EXPLORATION VALUE
EPSILON_MIN = 0.1				# MINIMUM EXPLORATION VALUE
EPSILON = 1.0					# INITIAL EXPLORARION VALUE
GREEDY_FACTOR = 0.000001		# EXPLOARTION DECAY RATE
EXPLORATION_FRAMES = 5000		# MAX EXPLORATION FRAMES
MAX_STEPS = 10000				# MAX NUMBER OF FRAMES PER EPISODE
GAMMA = 0.99					# REWARD DISCOUNT FACTOR
BATCH_SIZE = 32					
LEARNING_RATE = 0.0001
FRAMES_PER_EPOCH = 5000
MAX_REWARD = 200

is_image_input = False
save_model = False

class Agent(Model):
	def __init__(self, action, image=True):
		super(Agent, self).__init__()
		self.image = image
		initializer = tf.keras.initializers.VarianceScaling(scale=2)
		# Create Model
		if image:
			self.conv1 = Conv2D(16, 8, strides=4, activation='relu', kernel_initializer=initializer)
			self.conv2 = Conv2D(32, 4, strides=2, activation='relu', kernel_initializer=initializer)
			self.flatten = Flatten()
		self.d1 = Dense(256, activation='relu', kernel_initializer=initializer)
		self.d2 = Dense(action, activation='linear', kernel_initializer=initializer)

	def call(self, x):
		if self.image:
			x = self.conv1(x)
			x = self.conv2(x)
			x = self.flatten(x)
		x = self.d1(x)
		return self.d2(x)

# Create env for test
env = create_env(ENVIRONMENT)

ACTIONS = env.action_space.n

model = Agent(ACTIONS, is_image_input)
target_model = Agent(ACTIONS, is_image_input)

experiance_replay = deque(maxlen=REPLAY_MEMORY)

loss_function = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)#, clipnorm=1.0)
training_loss = tf.keras.metrics.Mean(name='training_loss')

frame_count = 0
count = 0
epoch = 1
reward_history = 0
episode_count = 0
loss = 0

print('Epoch {}'.format(epoch))
while True:
	state = np.array(env.reset())
	episode_count += 1
	for i in range(1, MAX_STEPS):
		# env.render()
		if get_random_num(0, 100, 0.01) <= EPSILON or frame_count < EXPLORATION_FRAMES:
			action = get_random_num(0, ACTIONS)
		else:
			state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
			state_tensor = tf.expand_dims(state_tensor, axis=0)
			pred = model(state_tensor, training=False)
			action = np.argmax(pred)

		next_state, reward, done, info = env.step(action)
		next_state = np.array(next_state)
		frame_count += 1
		count += 1
		reward_history += reward

		if EPSILON > EPSILON_MIN:
			EPSILON -= (EPSILON_MAX - EPSILON_MIN) * GREEDY_FACTOR
		else:
			EPSILON = EPSILON_MIN
		experiance_replay.append((state, action, reward, next_state, done))
		state = next_state
		## Training part
		## type casting of state to tensor is required 
		## (model takes tensor as input)

		if frame_count % UPDATE_FREQ == 0 and len(experiance_replay) > BATCH_SIZE:
			deque_len = len(experiance_replay)
			indices = [get_random_num(0, deque_len) for i in range(BATCH_SIZE)]

			replay_sample = [experiance_replay[i] for i in indices]
			state_sample = np.array([replay_sample[i][0] for i in range(BATCH_SIZE)])
			action_sample = [replay_sample[i][1] for i in range(BATCH_SIZE)]
			reward_sample = np.array([replay_sample[i][2] for i in range(BATCH_SIZE)])
			next_state_sample = np.array([replay_sample[i][3] for i in range(BATCH_SIZE)])
			done_sample = tf.convert_to_tensor([replay_sample[i][4] for i in range(BATCH_SIZE)], dtype=tf.float32)

			
			expected_reward = target_model(next_state_sample, training=False)
			q_value = reward_sample + GAMMA * tf.reduce_max(expected_reward, axis=1)
			
			# If the episode ends at this frame then set q value to -1
			q_value = q_value * (1 - done_sample) - done_sample

			mask = tf.one_hot(action_sample, ACTIONS)
			# expected future reward = maximum q value of next state
			# Q value = reward + discount factor * expected future reward
			with tf.GradientTape() as tape:
				pred = model(state_sample, training=True)
				q_action = tf.reduce_sum(tf.multiply(pred, mask), axis=1)
				loss = loss_function(q_value, q_action)
			gradients = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		if frame_count % UPDATE_TARGET == 0:
			target_model.set_weights(model.get_weights())

		if show_progress(count, FRAMES_PER_EPOCH, training_loss(loss)):
			average_reward = reward_history / episode_count
			print('Average epoch reward: {}'.format(average_reward))
			run_demo(ENVIRONMENT, target_model)
			reward_history = 0
			count = 0
			episode_count = 0
			# frame_count = 0

			if save_model and epoch > 4:
				parent_dir = 'Saved_Model/'
				model_dir = parent_dir + ENVIRONMENT + '_model_{}'.format(epoch)
				target_model_dir = parent_dir + ENVIRONMENT + '_target_model_{}'.format(epoch)
				os.mkdir(model_dir)
				os.mkdir(target_model_dir)

				model.save(model_dir, save_format="tf")
				target_model.save(target_model_dir, save_format="tf")
				
			if average_reward > MAX_REWARD:
				sys.exit()
			epoch += 1
			print('Epoch {}'.format(epoch))

		if done:
			break
env.close()