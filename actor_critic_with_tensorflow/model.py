import os  # handle file joining operation for model
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense  # our layers


# keras.Model в скобках значит, что наш класс наследуется от базового класса
class ActorCriticNetwork(keras.Model):
    # define our constructor
    # n_actions - количество действий
    # fc1_dims, fc2_dims - задаем размеры по умолчанию первого и второго слоев
    # name - имя файла для чекпоинтов
    # chkpt_dir - путь до директории, чтобы хранить файлы с чекпоинтами
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
                 name='actor_critic', chkpt_dir='D:\gui-test\checkpoints_for_actor_critic'):
        super(ActorCriticNetwork, self).__init__()

        # saving our parameters
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, self.model_name + '_ac')

        # Define our layers
        # fully connected dense layers
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')

        # value function, single valued with None activation
        self.v = Dense(1, activation=None)

        # our policy pi, policy it is just a probability distribution
        # assign probability for each action
        self.pi = Dense(n_actions, activation='softmax')

    # our call function - really it is feed forward function
    def call(self, state, **kwargs):
        value = self.fc1(state)
        value = self.fc2(value)

        # get our value function and our policy pi
        v = self.v(value)
        pi = self.pi(value)

        return v, pi

    def get_config(self):
        pass