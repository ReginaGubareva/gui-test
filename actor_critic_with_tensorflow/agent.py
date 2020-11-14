import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from actor_critic_with_tensorflow.model import  ActorCriticNetwork


class Agent:
    # alpha - learning rate
    # gamma -
    # n_actions - number of actions
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.n_actions = n_actions

        # variable keep the last action we took
        self.action = None

        # action_space for random action selection
        self.action_space = [i for i in range(self.n_actions)]

        # our model
        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)

        # to compile our model
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    # the basic functionality to choose an action
    # observation - current state of the environment
    def choose_action(self, observation):
        print('CHOOSE ACTION')
        
        state = tf.convert_to_tensor([observation])
        print('observation converted to tensor ', state)

        # getting probability
        _, probs = self.actor_critic(state)
        print('probs from actor_critic(state)', probs)

        # Then we can use prob output to feed it to the actual tensorflow
        # probabilities categorical distribution and the we use it to select
        # an action by sampling that distribution ang getting a log probability
        # of selecting that sample
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        print('action_probabilties: ', action_probabilities)

        # actual action will be a sample of the distribution action_probabilities
        action = action_probabilities.sample()
        print('get action as action_probabilities.sample()', action)
        log_prob = action_probabilities.log_prob(action)
        print('log_prob from action', log_prob)

        # for action to be selected we save in action variable
        self.action = action

        # return numpy version of our action because our action is tensorflow tensor
        # and it is not compatible with the open ai gym
        # we take a numpy array and get the zeroth element of that because added in batch
        # dimension for compatibility with our neural network
        print('return from choose action is numpy version of Action: ', action.numpy()[0])
        return action.numpy()[0]

    # a couple functions to save and load models
    def save_models(self):
        # it will save the weights of the network to the checkpoint file
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    # we do inverse operation to load models
    def load_models(self):
        print('Loading models')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    # main functionality to learn
    # state_ - new state
    # done - terminal flag
    def learn(self, state, reward, state_, done):
        print('LEARN')
        # the first we convert each to tensorflow tensors
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        print('state: ', state)
        print('state_: ', state_)
        print('reward: ', reward)

        # here we calculate our actual gradients using something called
        # the gradient tape
        # persistent = True is experimental parameter
        with tf.GradientTape(persistent=True) as tape:
            # we want to feed our state and new state through actor_critic network
            # and get back our quantities of interest
            # that is for calculation our delta
            state_value, probs = self.actor_critic(state)
            print('state_value after acotr_critic(state): ', state_value)
            print('probs after acotr_critic(state): ', probs)
            state_value_, _ = self.actor_critic(state_)
            print('state_value_ after acotr_critic(state): ', state_value_)

            # for the calculation of our loss we have to get rid
            # of that batch dimension, so we have to squeeze these two parameters
            state_value = tf.squeeze(state_value)
            print('state_value squeeze: ', state_value)
            state_value_ = tf.squeeze(state_value_)
            print('state_value_ squeeze: ', state_value_)

            # we need our action probabilities for the calculation of the log prob
            action_probs = tfp.distributions.Categorical(probs=probs)
            print('action_probs (distributions.Categorical): ', action_probs)
            log_prob = action_probs.log_prob(self.action)
            print('log_prob of action: ', log_prob)

            # count delta
            # this is a reward + gamma multiplied by the value of the new state
            # the value of the terminal state is zero, no returns, no rewards
            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
            print('delta: ', delta)
            actor_loss = -log_prob * delta
            print('actor_loss: ', actor_loss)
            critic_loss = delta ** 2
            print('critic_loss: ', critic_loss)
            total_loss = actor_loss + critic_loss
            print('total_loss: ', total_loss)

        # calculate gradients
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        print('gradient: ', gradient)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))
