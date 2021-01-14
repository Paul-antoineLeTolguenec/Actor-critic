import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np
import random
import os
from collections import namedtuple

Step = namedtuple('Step', ['state', 'log_prob', 'critic', 'reward'])


class ActorCritic(tf.keras.Model):
	
	def __init__(self, num_actions: int):
		super().__init__()
		num_hidden_units=40
		
		self.common_actor = layers.Dense(num_hidden_units, activation="relu",kernel_initializer=initializers.RandomNormal(stddev=0.01),bias_initializer=initializers.Zeros())
		self.common_critic = layers.Dense(num_hidden_units, activation="relu",kernel_initializer=initializers.RandomNormal(stddev=0.01),bias_initializer=initializers.Zeros())

		self.actor = layers.Dense(num_actions, activation="softmax",kernel_initializer=initializers.RandomNormal(stddev=0.01),bias_initializer=initializers.Zeros())

		self.critic = layers.Dense(1,activation="relu",kernel_initializer=initializers.RandomNormal(stddev=0.01),bias_initializer=initializers.Zeros())

	def call(self, inputs: tf.Tensor):
		x = self.common_actor(inputs)
		y = self.common_critic(inputs)

		return self.actor(x), self.critic(y)

class ReplayMemory(object):
	
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
	
	def push(self, event):
		self.memory.append(event)
		if len(self.memory) > self.capacity:
			del self.memory[0]
	
	def sample(self, batch_size):
		samples = zip(*random.sample(self.memory, batch_size))
		return map(lambda x: Variable(torch.cat(x, 0)), samples)

class Dqn():
	
	def __init__(self, input_size, nb_action, gamma, n_step=5):
		self.gamma = gamma
		self.n_step=5
		self.reward_window = []
		self.model = ActorCritic(nb_action)
		self.memory = ReplayMemory(self.n_step)
		# Configuration parameters for the whole setup
		self.eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.
		self.optimizer = keras.optimizers.RMSprop(learning_rate=0.006)
		self.huber_loss = keras.losses.Huber()
		self.actor_losses = []
		self.critic_losses = []
		self.rewards_history = []
		self.nb_action=nb_action
		self.running_reward = 0
		self.episode_count = 0
		self.last_state = tf.zeros([input_size])
		self.last_state= tf.expand_dims(self.last_state, 0)
		self.last_log_prob=0
		self.last_critic= 0
		self.last_action = 0
		self.last_reward = 0
	
	''' def select_action(self, state):
		probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
		action = probs.multinomial(num_samples=1)
		return action.data[0,0] '''
	
	''' def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
		outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
		next_outputs = self.model(batch_next_state).detach().max(1)[0]
		target = self.gamma*next_outputs + batch_reward
		td_loss = F.smooth_l1_loss(outputs, target)
		self.optimizer.zero_grad()
		td_loss.backward()
		self.optimizer.step() '''
	
	def update(self, reward, new_signal,tape):
    		#convert to tensor
		state = tf.convert_to_tensor(new_signal)
		state = tf.expand_dims(state, 0)
	
		
		# Predict action probabilities and estimated future rewards
		# from environment state
		action_probs, critic_value = self.model(state)
		
		#self.critic_value_history.append(critic_value[0, 0])
		
		# Sample action from action probability distribution
		action = np.random.choice(self.nb_action, p=np.squeeze(action_probs))
		log_prob=tf.math.log(action_probs[0, action])
		#self.action_probs_history.append(log_prob)


		#Memory update
		self.memory.push(Step(state=self.last_state, log_prob=self.last_log_prob, critic=self.last_critic, reward=reward))
  
		#learning process

		if self.memory.memory[0].log_prob!=0 and len(self.memory.memory)>=self.n_step:
			cumul_reward=0
			for step in reversed(self.memory.memory):
    				cumul_reward+=self.gamma*cumul_reward + step.reward
        
        
			diff = cumul_reward - self.memory.memory[0].critic.numpy()
			self.actor_losses.append(-(self.memory.memory[0].log_prob * diff))  # actor loss

			# The critic must be updated so that it predicts a better estimate of
			# the future rewards.
			self.critic_losses.append(
				self.huber_loss(tf.expand_dims(self.memory.memory[0].critic, 0), tf.expand_dims(cumul_reward, 0))
			)
			self.memory.memory=[]
			
			if len(self.actor_losses)>=2:
    			
				# Backpropagation
				loss_value = 0.5*sum(self.actor_losses) + sum(self.critic_losses)
				grads = tape.gradient(loss_value, self.model.trainable_variables)
				grads, _ = tf.clip_by_global_norm(grads, 10000.0)
				self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
				self.actor_losses = []
				self.critic_losses = []
			
				
		
		self.last_critic=critic_value[0, 0]
		self.last_log_prob=log_prob
		self.last_action = action
		self.last_state = state
		self.last_reward = reward
		self.reward_window.append(reward)
		if len(self.reward_window) > 1000:
			del self.reward_window[0]
		return action
	
	def score(self):
		return sum(self.reward_window)/(len(self.reward_window)+1.)
	
	def save(self):
		self.model.save('model')

	
	def load(self):
		if os.path.isfile('model'):
			print("=> loading checkpoint... ")
			self.model=keras.models.load_model('model')
			#self.optimizer.load_state_dict(checkpoint['optimizer'])
			print("done !")
		else:
			print("no checkpoint found...")

