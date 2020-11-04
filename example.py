import gym
from time import sleep
from IPython.display import clear_output

env = gym.make("Taxi-v3").env
##{action: [(probability, nextstate, reward, done)]}
# var = env.P[328]
# print(var)


env.s = 328

# Устанавливаем в ноль количество итераций, штрафы и вознаграждение,
epochs = 0
penalties, reward = 0, 0

frames = []

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Каждый отображенный кадр помещаем в словарь для анимации
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
    }
    )

    epochs += 1

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


# Выводим все возможные действия, состояния, вознаграждения
def func(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


func(frames)
