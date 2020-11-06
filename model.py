from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init


# Сейчас алгоритм принимает на вход num_inputs, что соответствует
# env.observation_space и action_space. Action_space - это простой
# массив действий, observation_space нужно заменить на state(screenshot)

# Это мозги A3C

class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3Clstm, self).__init__()

        # Применяет двумерную свертку к входному сигналу,
        # состоящему из нескольких входных плоскостей.
        # Conv2d input arguments:
        # num_inputs = Number of channels in the input image
        # out_channels = Number of channels produced by the convolution
        # kernel_size (int or tuple) – Size of the convolving kernel
        # stride (int or tuple, optional) – Stride of the convolution. Default: 1
        # padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)  # first convolution

        # Применяет максимальное 2D-объединение к входному сигналу,
        # состоящему из нескольких входных плоскостей.
        # kernel_size – the size of the window to take a max over
        # stride – the stride of the window. Default value is kernel_size
        # padding – implicit zero padding to be added on both sides
        # dilation – a parameter that controls the stride of elements in the window
        # return_indices – if True, will return the max indices along with the outputs.
        # Useful for torch.nn.MaxUnpool2d later
        # ceil_mode – when True, will use ceil instead of floor to compute the output shape
        self.maxp1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)  # second convolution
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)  # third convolution
        self.maxp3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # fourth convolution
        self.maxp4 = nn.MaxPool2d(2, 2)

        # A long short-term memory (LSTM) cell
        # input of shape (batch, input_size): tensor containing input features
        # h_0 of shape (batch, hidden_size): tensor containing the initial
        # hidden state for each element in the batch.
        # c_0 of shape (batch, hidden_size): tensor containing the initial
        # cell state for each element in the batch.
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        self.lstm = nn.LSTMCell(1024, 512)

        # getting the number of possible actions
        num_outputs = action_space.n

        # Applies a linear transformation to the incoming data
        # Linear(in_features=512, out_features=1, bias=True)
        self.critic_linear = nn.Linear(512, 1)  # full connection of the critic: output = V(S)
        self.actor_linear = nn.Linear(512, num_outputs) # full connection of the actor: output = Q(S,A)

        # Инициализация весов для CNN(это свертка входного изображения и
        # слоев, которые создали выше)
        self.apply(weights_init)  # initilizing the weights of the model with random weights

        # Return the recommended gain value for the
        # given nonlinearity function. The values are as follows
        # relu = sqrt(2). Функция просто возвращает приближенное значение.
        relu_gain = nn.init.calculate_gain('relu')

        # Инициализация весов
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        # setting the standard deviation of the actor tensor of weights to 0.01
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0) # initializing the actor bias with zeros

        # setting the standard deviation of the critic tensor of weights to 0.01
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)# initializing the critic bias with zeros

        # initializing the lstm bias with zeros
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # setting the module in "train" mode to activate the dropouts and batchnorms
        # Batchnorms - пакетная нормализация, для ускорения обучения
        self.train()

    def forward(self, inputs):
        # getting separately the input images to the tuple (hidden states, cell states)
        inputs, (hx, cx) = inputs

        # forward propagating the signal from the input images to the 1st convolutional layer
        x = F.relu(self.maxp1(self.conv1(inputs)))

        # forward propagating the signal from the 1st convolutional layer
        # to the 2nd convolutional layer
        x = F.relu(self.maxp2(self.conv2(x)))

        # forward propagating the signal from the 2nd convolutional layer
        # to the 3rd convolutional layer
        x = F.relu(self.maxp3(self.conv3(x)))

        # forward propagating the signal from the 3rd convolutional layer
        # to the 4th convolutional layer
        x = F.relu(self.maxp4(self.conv4(x)))

        # flattening the last convolutional layer into this 1D vector x
        x = x.view(x.size(0), -1)

        # the LSTM takes as input x and the old hidden & cell states
        # and ouputs the new hidden & cell states
        hx, cx = self.lstm(x, (hx, cx))

        # getting the useful output, which are the hidden states (principle of the LSTM)
        x = hx

        # returning the output of the critic (V(S)), the output of the actor (Q(S,A)),
        # and the new hidden & cell states ((hx, cx))
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
