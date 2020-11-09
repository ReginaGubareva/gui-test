from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(object):
    # Инициализируем агента
    def __init__(self, model, args, state):
        self.model = model
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1

    def action_train(self):

        # self.state.unsqueeze(0) - возвращает тензор размерностью один
        # Variable() - is a wrapper around a PyTorch Tensor,
        # and represents a node in a computational graph
        value, logit, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        print('Logit: ', logit)

        # Applied softmax function
        # Применяется ко всем фрагментам вдоль dim и масштабирует их так,
        # чтобы элементы лежали в диапазоне [0, 1] и прибавлялись к 1.
        # Подробнее читай, что такое softmax function
        prob = F.softmax(logit, dim=1)
        print('prob: ', prob)

        # Возвращает log(softmax(x))
        # log_probability is the probability for taking an action a_t in state s_t using policy pi
        log_prob = F.log_softmax(logit, dim=1)
        print('log_prob: ', log_prob)

        # Энтропия добавляется к целевой функции, потому что это помогает A3C лучше исследовать.
        # Добавление энтропии означает, что алгоритм оптимизации пытается поддерживать высокую энтропию,
        # и поэтому лежащее в основе распределение вероятности не должно сходиться
        # к неоптимальной детерминированной политике.
        # Русским языком, если у нас картина в которой действия абсолютно детерменированы, то алгоритм будет
        # плохо обучаться и если одинаковая вероятность у всех действий тоже будет плохо обучаться.
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)

        # Возвращает тензор, в котором каждая строка содержит индексы num_samples,
        # выбранные из полиномиального распределения вероятностей,
        # расположенного в соответствующей строке входного тензора.
        action = prob.multinomial(1).data
        print('action: ', action)

        # Gathers values along an axis specified by dim.
        # >>> t = torch.tensor([[1,2],[3,4]])
        # >>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
        # tensor([[ 1,  1],
        #         [ 4,  3]])
        log_prob = log_prob.gather(1, Variable(action))
        print('log_prob_gather: ', log_prob)

        # Здесь мы делаем действие и возвращем измененное состояние среды
        # награду, устанавливаем done, info - это лог
        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy())

        # torch.from_numpy(ndarray) → Tensor
        # Создает тензор из массива,в данном случае состояние должно возвращаться
        # как массив
        # float() - преобразует в тип float
        self.state = torch.from_numpy(state).float()

        # Здесь все итак ясно
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    # Это повторение функции action train, пока не ясно в чем отличие
    def action_test(self):
        # to avoid storing the intermediate tensors
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:

                    # Variables wrap a Tensor
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(
                            torch.zeros(1, 512).cuda())
                        self.hx = Variable(
                            torch.zeros(1, 512).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 512))
                    self.hx = Variable(torch.zeros(1, 512))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()

        # TODO: change env to screenshot and step to some actions with screenshot
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
