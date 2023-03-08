from simulation import sim
import os
import time
from tqdm import tqdm

data_num=20000

print('Program started')
sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 2000, 1)

if clientID != -1:
    print('Connected to remote API server')
# enable the synchronous mode on the client:

sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)

#create folder
if not os.path.exists('roll_data'):
    os.makedirs('roll_data')

for i in tqdm(range(data_num)):
    init_dir='roll_data/img'+str(i)
    if not os.path.exists(init_dir):
        os.makedirs(init_dir)

#generate data
for i in tqdm(range(data_num)):
    f = open("roll_data/a.txt", 'w')
    f.write(str(i))
    f.close()

    # start simulation
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    time.sleep(2.5)
    # stop simulation
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    time.sleep(0.5)