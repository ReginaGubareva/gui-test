from result.agent import Agent
from result.environment import Environment

env = Environment()
chekpoint_file = open(fr"tmp\trained_model", 'r')
checkpoints = []
scores = []
index_max = 0
max = 0.0
for i in chekpoint_file:
    checkpoints.append(i)
    checkpoint = i.split('; ')
    for j in range(len(checkpoint)):
        if j == (len(checkpoint) - 1):
            if float(checkpoint[j]) > max:
                max = float(checkpoint[j])
                index_max = checkpoint
            # scores.append(float(j))
print('index:', index_max, 'max', max)

for i in range(len(index_max) - 1):
    array = index_max[i].split()
    x_coords = int(array[1].replace('[', '').replace(',', ''))
    y_coords = int(array[2].replace(']', ''))
    coords = [x_coords, y_coords]
    print('action:', array[0], 'coords:', coords)
    observation_, reward, done, counter = env.step(array[0], coords, counter=0)
    counter += 1
