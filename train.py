from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from utils import ensure_shared_grads
from model import A3Clstm
from agent import Agent
from torch.autograd import Variable
from selenium import webdriver
import pyautogui


# нужно изменить аргументы, state мы не должны получать функцию
# мы должны менять его внутри функции
def train(rank, args, shared_model, optimizer, action_space):
    ptitle('Training Agent: {}'.format(rank))

    # this set gpu from command args --gpu-ids 0 1 2 3
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    # manual_seed() - генератор случайных чисел
    # аргумент - это число, с которого стартует генератор
    torch.manual_seed(args.seed + rank)

    # настройка gpu и добавление оптимизаторв
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    # Открываем наш сайт
    driver = webdriver.Chrome(fr'D:\chromedriver.exe')
    driver.get('https://digital.sberbank.kz/customer/login')

    # env.seed(args.seed + rank)
    agent = Agent(None, args, None)
    agent.gpu_id = gpu_id
    # Здесь мы создаем модель, первый аргумент - это количество каналов в изображении
    # в нашем случае RGB = 3, action space - набор совершаемых действий
    agent.model = A3Clstm(3, action_space)

    agent.state = pyautogui.screenshot()
    agent.state = torch.from_numpy(agent.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            agent.state = agent.state.cuda()
            agent.model = agent.model.cuda()
    agent.model.train()
    agent.eps_len += 2
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                agent.model.load_state_dict(shared_model.state_dict())
        else:
            agent.model.load_state_dict(shared_model.state_dict())
        if agent.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    agent.cx = Variable(torch.zeros(1, 512).cuda())
                    agent.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                agent.cx = Variable(torch.zeros(1, 512))
                agent.hx = Variable(torch.zeros(1, 512))
        else:
            agent.cx = Variable(agent.cx.data)
            agent.hx = Variable(agent.hx.data)

        # основную тренировку делает агент
        # TODO: неправильно передается num_steps
        for step in range(args.num_steps):
            agent.action_train()
            if agent.done:
                break

        # TODO: change environment to screenshot
        if agent.done:
            # здесь мы должны получать новое состояние
            # для этого выясним как происходит сброс среды
            # так же нужно учесть, что среда должна возвращать награду
            # функция reset() возвращает среду в первоначальное состояние,
            # перед началом каждого нового эпизода, для каждой env - это происходит по разному
            # в нашем алгоритме нужно получаеть новое состояние
            # state = agent.env.reset()
            state = pyautogui.screenshot()

            # вот здесь на вход должен приходить массив from_numpy(ndarray), который преобразовывается
            # в тензор, потом все значения тензора приводятся к типу float. Вот здесь сложно) нужно
            # преобразовать картинку в ndarray, какую информацию оставить неясно
            agent.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    agent.state = agent.state.cuda()

        R = torch.zeros(1, 1)
        if not agent.done:
            value, _, _ = agent.model((Variable(agent.state.unsqueeze(0)),
                                       (agent.hx, agent.cx)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        agent.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(agent.rewards))):
            R = args.gamma * R + agent.rewards[i]
            advantage = R - agent.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = agent.rewards[i] + args.gamma * \
                      agent.values[i + 1].data - agent.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                          agent.log_probs[i] * \
                          Variable(gae) - 0.01 * agent.entropies[i]

        agent.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(agent.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        agent.clear_actions()
        return
