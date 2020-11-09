from __future__ import print_function, division
import os

import description

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from utils import read_config
from model import A3Clstm
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
import time

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=32,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='Pong-v0',
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

# This is launch command:
# python description.py --env PongDeterministic-v4 --workers 32 --gpu-ids 0 1 2 3 --amsgrad True


# CUDA – это платформа для параллельных вычислений и модель API,
# созданная компанией Nvidia

if __name__ == '__main__':

    # читаем аргументы из команды запуска
    args = parser.parse_args()

    # устанавливает начальное число для генератора случайных чисел
    torch.manual_seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)

        # добаление мультпроцессорной обработки, тип spawn
        mp.set_start_method('spawn')

    # Читаем настройки из команды запуска
    # Эта строчка нам тоже не нужна, так как она отвечает
    # за то, чтобы считать парметры определенного env из json
    # setup_json = read_config(args.env_config)

    # Устанавливаем настройки среды по умолчани, нам весь остальной
    # код по созданию среды не нужен
    # env_conf = setup_json["Default"]
    # for i in setup_json.keys():
    #     if i in args.env:
    #         env_conf = setup_json[i]
    # env = atari_env(args.env, env_conf, args)

    # shared_model is the model shared by the different agents (different threads in different cores)
    # Заменим аргументы в a3clstm, вместо env.observation_space будем передавать
    # наше состояние, вместо env.action_space - передадим массив действий
    # shared_model = A3Clstm(env.observation_space.shape[0], env.action_space)
    shared_model = A3Clstm(3, description.action_space)

    # Здесь происходит что-то важное
    if args.load:
        saved_state = torch.load(
            '{0}{1}.dat'.format(args.load_model_dir, args.env),
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    # Добавляем оптимизаторы
    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    # Этот вызов создает один процесс для нашей функции,  which will process
    # который будет обрабатывать целевую функцию target-function с определенными
    # аргументами. Это вопрос оптимизации ресурсов процессора,
    # который не имеет отношения к самому обучению
    # выяснить почему аргументом не может быть state
    # allowing to create the 'test' process with some arguments 'args' passed to the
    # 'test' target function - the 'test' process doesn't update the shared model
    # but uses it on a part of it - torch.multiprocessing.Process runs a function in an independent thread
    p = mp.Process(target=test, args=(args, shared_model))
    p.start()
    processes.append(p)
    time.sleep(0.1)

    # аргумент workers означает то, сколько процессов нужно использовать
    # по умолчанию значение workers = 32
    # здесь мы для каждого процесса передаем параметры и запускаем его
    for rank in range(0, args.workers):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()
