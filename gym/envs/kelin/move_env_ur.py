# Environment for only training UR5 to move to target end effector point

from numpy import random
import pybullet 
import numpy as np
import time
import pybullet_data
from scipy.spatial.transform import Rotation as R
import os
# import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from random import choice

#======= Load contact points
def load_contact_pt(contact_point_fpath, row_index):
    cp = np.load(contact_point_fpath)  #npy file [10x9] with 10 sets of possible contact points combinations (3 points with xyz)
    cp_chosen = cp[row_index]

    cp_list = np.array(cp_chosen).reshape(-1,3)
    motor_control_ruth(cp_list)

#======= Visualize contact points as spheres
def visualize_contact_pt(point_pos, rgba, _robotScale):
    colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=0.01 * _robotScale)
    visualShapeId = -1
    sphereA = pybullet.createMultiBody(0.01, colSphereId, visualShapeId, point_pos)

    # Disable Collisions between links
    pybullet.setCollisionFilterGroupMask(sphereA, -1, 1, 0)
    con2 = pybullet.createConstraint(parentBodyUniqueId=sphereA,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=pybullet.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=point_pos)

    pybullet.changeVisualShape(sphereA, -1, rgbaColor=rgba)
    

#===== Join RUTH parallel links together, initialize motor positions to 0
def initialize_RUTH(_robot, _RUTH_linkNameToID, _RUTH_jointNameToID):
    link2CoM = [-0.0138504507471105, 0.00298913196612882, 0.0239892494327788]
    link2CoM2 = np.array([link2CoM[0], 0, link2CoM[2]])
    link2_Joint = -link2CoM2 + 0.07*link2CoM2/np.linalg.norm(link2CoM2)

    link4CoM = [0.0139032099348594, -0.00302765430432597, 0.0240810659954725]
    link4CoM2 = np.array([link4CoM[0], 0, link4CoM[2]])
    link4_Joint = -link4CoM2 + 0.07*link4CoM2/np.linalg.norm(link4CoM2)

    link2_info = pybullet.getLinkState(_robot, _RUTH_linkNameToID['Link_2'])
    link4_info = pybullet.getLinkState(_robot, _RUTH_linkNameToID['Link_4'])
    CoM_diff = np.array(link4_info[0]) - np.array(link2_info[0])
    link2_Joint = link2_Joint + np.array([0, CoM_diff[2], 0])

    con1 = pybullet.createConstraint(parentBodyUniqueId=_robot,
                        parentLinkIndex= _RUTH_linkNameToID['Link_2'],
                        childBodyUniqueId=_robot,
                        childLinkIndex= _RUTH_linkNameToID['Link_4'],
                        jointType= pybullet.JOINT_POINT2POINT,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[link2_Joint[0], link2_Joint[1], link2_Joint[2]],
                        childFramePosition=[link4_Joint[0], link4_Joint[1], link4_Joint[2]])
    
    initPos1 = 1*np.pi/100000
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Link_1'], pybullet.POSITION_CONTROL, -initPos1, force=1000)
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Link_3'], pybullet.POSITION_CONTROL, initPos1, force=1000)

#===== Joint motor control on UR5
def motor_control_ur5(_robot, _ur5JointNameToID, _ur5_joint_name, _ur5_values):
    for index, ur5_control in enumerate(_ur5_values):   #ur5_control = ur5_values[index]
        pybullet.setJointMotorControl2(_robot, _ur5JointNameToID[_ur5_joint_name[index+1]], pybullet.POSITION_CONTROL, _ur5_values[index], force=1000)

#===== Joint motor control on RUTH
def motor_control_ruth(_robot, _RUTH_jointNameToID, _ruth_base_motors):
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Link_1'], pybullet.POSITION_CONTROL, _ruth_base_motors[0], force=1000)
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Link_3'], pybullet.POSITION_CONTROL, _ruth_base_motors[1], force=100)  

    # Calculate relative individual finger joint movements
    finger_angles = [0, 0, 0]
    finger_pos_plus = 0.12745044
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_1A'], pybullet.POSITION_CONTROL, -finger_angles[0], force=1000)
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_2A'], pybullet.POSITION_CONTROL, finger_angles[1], force=100)  
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_3A'], pybullet.POSITION_CONTROL, finger_angles[2], force=100)
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_1B'], pybullet.POSITION_CONTROL, -0.55-_ruth_base_motors[2]-finger_pos_plus, force=1000)
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_2B'], pybullet.POSITION_CONTROL, -0.55-_ruth_base_motors[2]-finger_pos_plus, force=100)
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_3B'], pybullet.POSITION_CONTROL, 0.65+_ruth_base_motors[2]+finger_pos_plus, force=100) 
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_1C'], pybullet.POSITION_CONTROL, 1-_ruth_base_motors[2]-finger_pos_plus, force=100)   
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_2C'], pybullet.POSITION_CONTROL, 1-_ruth_base_motors[2]-finger_pos_plus, force=1000)
    pybullet.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_3C'], pybullet.POSITION_CONTROL, -1+_ruth_base_motors[2]+finger_pos_plus, force=100)

#======= Config parameter panel
class pybulletDebug:
    def __init__(self, robot, ur5LinkNameToID, control_type='motors'):
        #Camera paramers to be able to yaw pitch and zoom the camera (Focus remains on the robot) 
        self.cyaw=90
        self.cpitch=-7
        self.cdist=0.66
        time.sleep(0.5)
        self.init_state = pybullet.getLinkState(robot, ur5LinkNameToID['ee_link'])
        self.init_pos = np.array(self.init_state[0])
        self.init_ori = np.array( pybullet.getEulerFromQuaternion(self.init_state[1]))

       
        if control_type == 'motors':
            self.control_type = 'motors'
            self.U1Id = pybullet.addUserDebugParameter("UR5 Shoulder Pan" , -3.14 , 3.14 , 0.)
            self.U2Id = pybullet.addUserDebugParameter("UR5 Shoulder Lift" , -3.14 , 3.14 , 0.)
            self.U3Id = pybullet.addUserDebugParameter("UR5 Elbow" , -3.14 , 3.14 , 0.)
            self.U4Id = pybullet.addUserDebugParameter("UR5 Wrist 1" , -3.14 , 3.14 , 0.)
            self.U5Id = pybullet.addUserDebugParameter("UR5 Wrist 2" , -3.14 , 3.14 , 0.)
            self.U6Id = pybullet.addUserDebugParameter("UR5 Wrist 3" , -3.14 , 3.14 , 0.)
    
        if control_type == 'xyz':
            self.control_type = 'xyz'
        #=== End effector pose xyz and orientation
            # self.U1Id = p.addUserDebugParameter("UR5_EE X" , -1.0 , 1.0 , self.init_pos[0]) 
            # self.U2Id = p.addUserDebugParameter("UR5_EE Y" , -1.0 , 1.0 , self.init_pos[1])
            # self.U3Id = p.addUserDebugParameter("UR5_EE Z" , 0.0 , 1.0 , self.init_pos[2])
            # self.U4Id = p.addUserDebugParameter("UR5_EE Rx" , -3.14 , 3.14 , self.init_ori[0])
            # self.U5Id = p.addUserDebugParameter("UR5_EE Ry" , -3.14 , 3.14 , self.init_ori[1])
            # self.U6Id = p.addUserDebugParameter("UR5_EE Rz" , -3.14 , 3.14 , self.init_ori[2])


        #=== RUTH parameters
        # Base Motor2 - decrease value for clockwise rotation
        # When both base links are near the center of the palm, the rotation is about |1 rad|
        # Fingers most closed position at tendon motor = 0.12 rad
        # self.Rm1Id = pybullet.addUserDebugParameter("RUTH Motor 1" , -1.0 , 1.57 , 0.) 
        # self.Rm2Id = pybullet.addUserDebugParameter("RUTH Motor 2" , -1.57 , 1.0 , 0.) 
        # self.RfId = pybullet.addUserDebugParameter("RUTH fingers" , -0.5 , 0.12 , 0.)

    
    def return_robot_states(self):
        ur5_values = [pybullet.getJointState(1,1)[0],pybullet.getJointState(1,2)[0],
                        pybullet.getJointState(1,3)[0],pybullet.getJointState(1,4)[0],
                        pybullet.getJointState(1,5)[0],pybullet.getJointState(1,6)[0]]
        return ur5_values 
    
import gym
from gym import spaces 
from gym.utils import seeding
import numpy as np

class MoveUr5RuthEnv(gym.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, version):
        #========= Initialize environment
        if version == 'GUI':
            physicsClient = pybullet.connect(pybullet.GUI)
            # p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
        elif version == 'DIRECT':
            physicsClient = pybullet.connect(pybullet.DIRECT) #non-graphical version

        
        # 
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        pybullet.setGravity(0,0,-9.81)
        pybullet.setRealTimeSimulation(0) #0: disable

        #===== Initialize plane and import robot
        FixedBase = False #if fixed no plane is imported
        if (FixedBase == False):
            floor = pybullet.loadURDF("plane.urdf")

        urdfDirectory = "/gym/envs/kelin/urdf/ur5.urdf" 
        robotPos = [0, 0, 0]
        robotScale = 1
        self.robot = pybullet.loadURDF(urdfDirectory,
                            robotPos,
                            pybullet.getQuaternionFromEuler([0, 0, 0]),
                            useFixedBase=0,
                            globalScaling=robotScale)

        #===== Check joint information from URDF file
        jointNum = pybullet.getNumJoints(self.robot)
        # print('\n\n', jointNum)
        for jt in range(jointNum):
            jointInfo = pybullet.getJointInfo(self.robot, jt)
            # print('\n\n',jointInfo)

        #===== Save UR5 joint and link information
        ur5JointNameToID = {}
        ur5LinkNameToID = {}
        ur5_joint_names = []
        ur5RevoluteID = []

        num_of_ur5_joints = 6

        for j in range(num_of_ur5_joints+2):
            info = pybullet.getJointInfo(self.robot, j)
            jointID = info[0]
            jointName = info[1].decode('UTF-8')
            jointType = info[2]
            ur5JointNameToID[jointName] = info[0]
            ur5LinkNameToID[info[12].decode('UTF-8')] = info[0]
            ur5RevoluteID.append(j)
            ur5_joint_names.append(jointName)



        # # ===== Save RUTH joint and link information
        # RUTH_jointNameToID = {} # Dictionary of RUTH joint and respective ID
        # RUTH_linkNameToID = {} # Dictionary of RUTH link and respective ID
        # RUTH_revoluteID = [] # array list of revolute joint ID?

        # for j in range(num_of_ur5_joints+2, pybullet.getNumJoints(robot)): #range 8-23. RUTH joints/links
        #     info = pybullet.getJointInfo(robot, j)
        #     jointID = info[0]
        #     jointName = info[1].decode('UTF-8')
        #     jointType = info[2]
        #     RUTH_jointNameToID[jointName] = info[0]
        #     RUTH_linkNameToID[info[12].decode('UTF-8')] = info[0]
        #     RUTH_revoluteID.append(j)

        # # print(RUTH_linkNameToID)

        # #===== Disable Collisions between links
        # for link in RUTH_linkNameToID:
        #     pybullet.setCollisionFilterGroupMask(robot, RUTH_linkNameToID[link], 1, 0)
        
        # for link in ur5LinkNameToID:
        #     pybullet.setCollisionFilterGroupMask(robot, ur5LinkNameToID[link], 1, 0)
        # # Disable collision between environments
        # pybullet.setCollisionFilterGroupMask(robot, -1, 1, 0) 
        # pybullet.setCollisionFilterGroupMask(floor, -1, 1, 0)

        # # Set RUTH parallel links
        # initialize_RUTH(robot, RUTH_linkNameToID, RUTH_jointNameToID)

        #======= Visualize contact points as spheres
        # contact_points = load_contact_pt(".\Contact_Points\Ruth\Ruth_Banana.npy", 1)
        # visualize_contact_pt(contact_points, robotScale)

        #=======Start simulation
        # self.RUTH_linkNameToID = RUTH_linkNameToID
        # self.RUTH_jointNameToID = RUTH_jointNameToID
        self.ur5JointNameToID = ur5JointNameToID
        self.ur5_joint_names = ur5_joint_names 
        self.ur5LinkNameToID = ur5LinkNameToID

        self.pybulletDebug1 = pybulletDebug(self.robot, self.ur5LinkNameToID)
        self._ur5_init = [0,np.pi/2,-np.pi/2,-np.pi/2,np.pi/2,0 ]
        
        self.version = version 

        # self.action_space = spaces.Box(np.array([-1.0,]*9), np.array([1.0,]*9))
        # self.observation_space = spaces.Box(np.array([0.0]*9), np.array([1.0]*9))
        self.action_space = spaces.Box(np.array([-1.0,]*6), np.array([1.0,]*6))
        self.observation_space = spaces.Box(np.array([0.0]*9), np.array([1.0]*9))
        self.goal_space = spaces.Box(np.array([0.0]*3), np.array([1.0]*3))
        
        self.max_delta_action = 0.08
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, flag=True, count=0):
        
        action = np.clip(action, -1, 1) 
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        action = action * self.max_delta_action 
        
        ur5_values = self.pybulletDebug1.return_robot_states() 
        ur5_values = action + ur5_values
        target = ur5_values
        motor_control_ur5(self.robot, self.ur5JointNameToID, self.ur5_joint_names, ur5_values)
        while(flag):
          pybullet.stepSimulation()
          real = self.get_obs()
          # real = np.asarray(real[3:])
          real = np.asarray(real[:6])
          err = np.sum((real-target)*(real-target))
          count = count+1
          #print(err)
          if err < 1e-3 or count==100:
              flag = False
        # time.sleep(1./20.)
        #print(target)
        obs = self.get_obs() 
        pos = self.get_pos()
        
        for index, ur5_control in enumerate(ur5_values):
            if ur5_control>=np.pi or ur5_control<=-np.pi:
                r = -10
                
        dist = np.linalg.norm(self.get_pos() - self.goal)     
        d = False 
        if dist < 0.01:
            r = 10
            d = True
        else:
            r = -dist
        # reward = - np.linalg.norm(self.get_pos() - self.goal) + penalty
        
        return obs, r, d, {}  

    def reset(self, flag=True, count=0):
        motor_control_ur5(self.robot, self.ur5JointNameToID, self.ur5_joint_names, self._ur5_init)
        target=np.array([0,np.pi/2,-np.pi/2,-np.pi/2,np.pi/2,0])
        self.reset_goal()
        while(flag):
          pybullet.stepSimulation()
          real = self.get_obs()
          real = np.asarray(real[:6])
          err = np.sum((real-target)*(real-target))
          count=count+1
          #print(err)
          if err < 1e-3 or count==100:
              flag = False
        pybullet.stepSimulation()
         
        state = self.get_obs() 
        return state 
    
    def reset_goal(self, goal=None): 
        x = random.uniform(0.3,0.5)
        y = random.uniform(-0.15,0.15)
        z = random.uniform(0.1,0.3)
        self.goal = np.array([x,y,z])
        # if goal is None:
        #     xyz = np.array([x,y,z])
        #     xyzd = random.rand(9)*0.1
        #     goal = np.array(xyz.tolist()*3) + xyzd 
        # if goal.shape != (9,):
        #     print("the shape of goal should be (9,)")
        # self.goal = goal 
        if self.version == "GUI":
            visualize_contact_pt(self.goal[:3], [1,0,0,1], 2)
            visualize_contact_pt(self.goal[3:6], [0,1,0,1], 2)
            visualize_contact_pt(self.goal[6:], [0,0,1,1], 2) 
        return self.goal 
    
    def point_transform(self):
        p = pybullet.getLinkState(self.robot,self.ur5LinkNameToID['ee_link'])[0]
        a, ur_joints = self.pybulletDebug1.return_robot_states()
        ur_joints[3] = ur_joints[3] + random.uniform(-np.pi/6, np.pi/6)
        ur_joints[4] = ur_joints[4] + random.uniform(-np.pi/6, np.pi/6)
        motor_control_ur5(self.robot, self.ur5JointNameToID, self.ur5_joint_names, ur_joints)
        for i in range(50):
            pybullet.stepSimulation()
        p = pybullet.getLinkState(self.robot,self.ur5LinkNameToID['ee_link'])[0]
        Q = pybullet.getLinkState(self.robot,self.ur5LinkNameToID['ee_link'])[1]
        R = pybullet.getMatrixFromQuaternion(Q)
        T_eb = np.array([[R[0],R[1],R[2],p[0]],[R[3],R[4],R[5],p[1]],[R[6],R[7],R[8],p[2]],[0,0,0,1]])
        goal_e1 = np.array([[self.goal[0]],[self.goal[1]],[self.goal[2]],[1]])
        goal_e2 = np.array([[self.goal[3]],[self.goal[4]],[self.goal[5]],[1]])
        goal_e3 = np.array([[self.goal[6]],[self.goal[7]],[self.goal[8]],[1]])
        goal_b1 = np.dot(T_eb,goal_e1).T[0,:3]
        goal_b2 = np.dot(T_eb,goal_e2).T[0,:3]
        goal_b3 = np.dot(T_eb,goal_e3).T[0,:3]
        cp_list = np.vstack((goal_b1,goal_b2,goal_b3))
        cp12 = (cp_list[1] - cp_list[0]) / np.linalg.norm(cp_list[1] - cp_list[0])
        cp13 = (cp_list[2] - cp_list[0]) / np.linalg.norm(cp_list[2] - cp_list[0])
        cp_normal = np.cross(cp12, cp13) / np.linalg.norm(np.cross(cp12, cp13))
        if cp_normal[2] < 0 : # if normal vector is downward
          cp_normal *= -1
        goal_b1 = goal_b1 + 0.1*cp_normal
        goal_b2 = goal_b2 + 0.1*cp_normal
        goal_b3 = goal_b3 + 0.1*cp_normal
        self.goal_b = np.hstack((goal_b1,goal_b2,goal_b3))
        if self.version == "GUI":
            visualize_contact_pt(self.goal_b[:3], [1,0,0,1], 2)
            visualize_contact_pt(self.goal_b[3:6], [0,1,0,1], 2)
            visualize_contact_pt(self.goal_b[6:], [0,0,1,1], 2) 
        cp_center = np.empty([3]) #center of the 3 contact points
        for i in range (3):
          cp_center[i] = sum(cp_list[:,i])/3
        cp_center = cp_center + 0.1*cp_normal
        orientation = np.array(Q)
        self.visualize_frame(orientation,cp_center)
            
    def visualize_frame(self, orien, shape_CoM):
        boxID = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents = [0.005,0.2,0.15])
        visualShapeId = -1

        bound_box = pybullet.createMultiBody(0.01, boxID, visualShapeId, [0.15,0.2,0.01])

        # Disable Collisions with external objects
        pybullet.setCollisionFilterGroupMask(bound_box, -1, 0, 0)
        pybullet.changeVisualShape(bound_box, -1, rgbaColor=[0,0,1,0.2])
        constraint = pybullet.createConstraint(parentBodyUniqueId=bound_box,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=pybullet.JOINT_FIXED,
                jointAxis=[0,0,0],
                parentFramePosition=[0,0,0],
                childFramePosition= shape_CoM,
                childFrameOrientation=orien)
        
    def get_obs(self):
        ur5_values = self.pybulletDebug1.return_robot_states()
        return np.concatenate((ur5_values, self.goal)).copy()

    
    def get_pos(self):
        #===========Get final fingertip position
        state = pybullet.getLinkState(self.robot, self.ur5LinkNameToID['ee_link'])
        return state[0]
    


#    def render(self, mode='human', **kwargs):
#        return self.env.render(mode, **kwargs)

    def render_frame(self, hw=15, imagesize=(300,300), camera_id=0, heatmap=None): 
        return None  

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# if __name__ == "__main__":
#     import gym
#     ro = gym.make("kelin-v0", version="DIRECT")
#     sta = ro.reset() 
#     ob = ro.get_obs()
#     print(ob)
#     for i in range(500):
#         a = ro.action_space.sample() #* 0 + 0.1
#         o, r, d, _ = ro.step(a)
#         # print(o)
#         if i % 20 == 0:
#             ro.reset() 
#         time.sleep(1./20.)
#     sta = ro.reset() 
#     ob = ro.get_obs()
#     print(ob)
#     pass 