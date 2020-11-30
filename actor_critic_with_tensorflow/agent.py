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
        print('observation', observation)
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        print('probs', action_probabilities)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        self.action = action
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
        # the first we convert each to tensorflow tensors
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        # here we calculate our actual gradients using something called
        # the gradient tape
        # persistent = True is experimental parameter
        with tf.GradientTape(persistent=True) as tape:
            # we want to feed our state and new state through actor_critic network
            # and get back our quantities of interest
            # that is for calculation our delta
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)

            # for the calculation of our loss we have to get rid
            # of that batch dimension, so we have to squeeze these two parameters
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            # we need our action probabilities for the calculation of the log prob
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            # count delta
            # this is a reward + gamma multiplied by the value of the new state
            # the value of the terminal state is zero, no returns, no rewards
            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2
            total_loss = actor_loss + critic_loss

        # calculate gradients
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))
