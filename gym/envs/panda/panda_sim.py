import time
import numpy as np
import math
import random

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
#rp = jointPositions

class PandaSim(object):
  def __init__(self, bullet_client, offset):
    
    self.bullet_client = bullet_client
    self.offset = np.array(offset)

    self.bullet_client.resetDebugVisualizerCamera(1.5,45,-45,[0,0,0])
    visualShapeId = self.bullet_client.createVisualShape(shapeType=self.bullet_client.GEOM_SPHERE,
                                    radius = 0.05,
                                    rgbaColor=[1, 0, 0, 1])
    self.ball = self.bullet_client.createMultiBody(baseMass=0,
                      baseCollisionShapeIndex=-1,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=[0,0,0],
                      useMaximalCoordinates=True)
    
    #print(self.bullet_client.getBasePositionAndOrientation(self.ball))
    #time.sleep(5)
    flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    #legos=[]
    #self.bullet_client.loadURDF("tray/traybox.urdf", [0+offset[0], 0+offset[1], -0.6+offset[2]], [-0.5, -0.5, -0.5, 0.5], flags=flags)
    #legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.5])+self.offset, flags=flags))
    #legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([-0.1, 0.3, -0.5])+self.offset, flags=flags))
    #legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.7])+self.offset, flags=flags))
    #sphereId = self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.6])+self.offset, flags=flags)
    #self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.5])+self.offset, flags=flags)
    #self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.7])+self.offset, flags=flags)
    orn=[-0.707107, 0.0, 0.0, 0.707107]
    self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
   
    index = 0
    for j in range(self.bullet_client.getNumJoints(self.panda)):
      
      self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
      info = self.bullet_client.getJointInfo(self.panda, j)
      jointName = info[1]
      jointType = info[2]
      if (jointType == self.bullet_client.JOINT_PRISMATIC):       
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
        index=index+1
      if (jointType == self.bullet_client.JOINT_REVOLUTE):
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
        index=index+1
    self.t = 0.
    
  def reset(self):
    pass

  def step(self, action):
    self.action = action
    #print(self.action)
    obs=[0,0,0,0,0,0,0]
    t = self.t
    self.t += 1./10.   

    for i in range(pandaNumDofs):
        self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, self.action[i],force=5 * 240.)
        getJoint = self.bullet_client.getJointState(self.panda,i)
        obs[i]=getJoint[0]
    
    current_pos = self.bullet_client.getLinkState(self.panda,7)

    goal_pos = np.array([random.uniform(-0.5,0.5),0.5+random.uniform(-0.5,0.5),random.uniform(-0.5,0.5)]) * 0 + [0.2,0.2,-0.2]
    self.bullet_client.resetBasePositionAndOrientation(self.ball,goal_pos,[0,0,0,1])
    
    
    
    #time.sleep(5)

    camera = self.bullet_client.getCameraImage(width=512, height=512)
    print(camera)
    return obs, current_pos, goal_pos, camera
    
  
