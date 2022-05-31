import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
#import pybullet_robots.panda.panda_sim as panda_sim
import panda_sim

import random


p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
p.setAdditionalSearchPath(pd.getDataPath())
#planeId = p.loadURDF("plane.urdf")

timeStep=1./10.
p.setTimeStep(timeStep)
p.setGravity(0,-9.8,0)

panda = panda_sim.PandaSim(p,[0,0,0])
action=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]

while (1):        
	obs,current_pos, goal_pos, camera = panda.step(action)
	#print(obs)
	p.stepSimulation()
	time.sleep(timeStep)
	action = [action[0]+random.uniform(-0.1,0.1),action[1]+random.uniform(-0.1,0.1),action[2]+random.uniform(-0.1,0.1),action[3]+random.uniform(-0.1,0.1),action[4]+random.uniform(-0.1,0.1),action[5]+random.uniform(-0.1,0.1),action[6]+random.uniform(-0.1,0.1)]
        
