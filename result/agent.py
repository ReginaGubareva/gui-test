import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from result.model import  ActorCriticNetwork
import random
from sklearn.preprocessing import normalize
import numpy as np

class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]
        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))


    def choose_action(self, observation):
        # action_space = ['click', 'type']
        # state = tf.keras.preprocessing.image.img_to_array(observation)
        state = self.convert_img_to_tensor(observation)
        state = tf.convert_to_tensor(state)
        _, probs = self.actor_critic(state)
        print('probs', probs)
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        print('probs', action_probabilities)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        self.action = action
        print('action', action.numpy()[0])
        if action.numpy()[0] == 0:
            return 'click'
        else:
            return 'type'

        # action = random.choice(action_space)
        # print('action', action.numpy()[0])


    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)


    def load_models(self):
        print('Loading models')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)


    def learn(self, state, reward, state_, done):
        # state = tf.convert_to_tensor([state], dtype=tf.float32)
        state = self.convert_img_to_tensor(state)
        state = tf.convert_to_tensor(state)

        state_ = self.convert_img_to_tensor(state_)
        state_ = tf.convert_to_tensor(state_)

        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)

            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2
            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))


    def convert_img_to_tensor(self, state):
        state = np.array(state)
        shape = state.shape
        normalized_metrics = normalize(state, axis=0, norm='l1')
        # flat_arr = state.ravel()
        # result_arr = []
        # for i in range(len(flat_arr)):
        #     print('arr[i]', flat_arr[i], float(flat_arr[i]) / float(255))
        #     result_arr.append(float(flat_arr[i]) / float(255))
        #
        # for i in range(len(result_arr)):
        #     print('result_arr', result_arr[i])

        return normalized_metrics
