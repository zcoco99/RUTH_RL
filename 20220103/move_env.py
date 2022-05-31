# Set up UR5 and RUTH for simulation
# This file simply tests UR5 and RUTH movement control
# 2021-Oct-23 @Xian Zhang
#=========================READ ME==========================
# When calling main(mode, timer), first parameter takes 0 or 1 for control mode:
# (0) for manual motor control, (1) to start automated testing
# timer parameter is how long you want the simulation to run for.
# Edit automated testing in function auto_joint_test(), by specifying which joints to start auto testing
# Final UR5 and RUTH joint states, final fingertip position and orientation are saved in array output_motor_fingertip




from numpy import random
import pybullet 
import numpy as np
import time
import pybullet_data
from scipy.spatial.transform import Rotation as R
from gym.envs.kelin.ruth_grasping_kinematics import ruthModel
from gym.envs.kelin.pybullet_object_models import ycb_objects
import os
# import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd

class FingerAngles:
    def __init__(self, robot, linkNameToID):
        self.robot = robot
        self.linkNameToID = linkNameToID
        Phal_1A_info = pybullet.getLinkState(self.robot, self.linkNameToID['Phal_1A'])
        Phal_2A_info = pybullet.getLinkState(self.robot, self.linkNameToID['Phal_2A'])
        Phal_3A_info = pybullet.getLinkState(self.robot, self.linkNameToID['Phal_3A'])
        P2 = np.array(Phal_1A_info[0][0:2])
        P3 = np.array(Phal_2A_info[0][0:2])
        P4 = np.array(Phal_3A_info[0][0:2])
        C = (P2+P3+P4)/3
        P2C = C-P2
        P3C = C-P3
        P4C = C-P4
        P2P3 = P3-P2
        P3P4 = P4-P3
        angle1 = np.arctan2(P2P3[0]*P2C[1]-P2P3[1]*P2C[0],P2P3[0]*P2C[0]+P2P3[1]*P2C[1])
        angle2 = np.arctan2(P2P3[0]*P3C[1]-P2P3[1]*P3C[0],P2P3[0]*P3C[0]+P2P3[1]*P3C[1])
        angle3 = np.arctan2(P3P4[0]*P4C[1]-P3P4[1]*P4C[0],P3P4[0]*P4C[0]+P3P4[1]*P4C[1])
        self.init_angles = [angle1, angle2, angle3]

    def calc_finger_angles(self):
        Phal_1A_info = pybullet.getLinkState(self.robot, self.linkNameToID['Phal_1A'])
        Phal_2A_info = pybullet.getLinkState(self.robot, self.linkNameToID['Phal_2A'])
        Phal_3A_info = pybullet.getLinkState(self.robot, self.linkNameToID['Phal_3A'])
        P2 = np.array(Phal_1A_info[0][0:2])
        P3 = np.array(Phal_2A_info[0][0:2])
        P4 = np.array(Phal_3A_info[0][0:2])
        C = (P2+P3+P4)/3
        P2C = C-P2
        P3C = C-P3
        P4C = C-P4
        P2P3 = P3-P2
        P3P4 = P4-P3
        angle1 = np.arctan2(P2P3[0]*P2C[1]-P2P3[1]*P2C[0],P2P3[0]*P2C[0]+P2P3[1]*P2C[1])
        angle2 = np.arctan2(P2P3[0]*P3C[1]-P2P3[1]*P3C[0],P2P3[0]*P3C[0]+P2P3[1]*P3C[1])
        angle3 = np.arctan2(P3P4[0]*P4C[1]-P3P4[1]*P4C[0],P3P4[0]*P4C[0]+P3P4[1]*P4C[1])
        angles = [angle1-self.init_angles[0], angle2-self.init_angles[1], angle3-self.init_angles[2]]
        return angles

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
    def __init__(self, control_type, robot, ur5LinkNameToID):
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
        self.Rm1Id = pybullet.addUserDebugParameter("RUTH Motor 1" , -1.0 , 1.57 , 0.) 
        self.Rm2Id = pybullet.addUserDebugParameter("RUTH Motor 2" , -1.57 , 1.0 , 0.) 
        self.RfId = pybullet.addUserDebugParameter("RUTH fingers" , -0.5 , 0.12 , 0.)

    
    def return_robot_states(self):
        # motor_positions = np.array([pybullet.readUserDebugParameter(self.Rm1Id), 
        #                             pybullet.readUserDebugParameter(self.Rm2Id), 
        #                             pybullet.readUserDebugParameter(self.RfId)]) 
        # # RUTH motor 1 [-1.57 , 3.14], motor 2 [-3.14 , 1.57],  RUTH finger bend [-0.5, 0.5]
        # ur5_values = np.array([pybullet.readUserDebugParameter(self.U1Id), 
        #         pybullet.readUserDebugParameter(self.U2Id), 
        #         pybullet.readUserDebugParameter(self.U3Id), 
        #         pybullet.readUserDebugParameter(self.U4Id), 
        #         pybullet.readUserDebugParameter(self.U5Id), 
        #         pybullet.readUserDebugParameter(self.U6Id)])
        ur5_values = [pybullet.getJointState(1,1)[0],pybullet.getJointState(1,2)[0],
                        pybullet.getJointState(1,3)[0],pybullet.getJointState(1,4)[0],
                        pybullet.getJointState(1,5)[0],pybullet.getJointState(1,6)[0]]
        motor_positions = [pybullet.getJointState(1,10)[0],pybullet.getJointState(1,15)[0],
                        -(pybullet.getJointState(1,13)[0]+0.12745044+0.55)]
        return motor_positions, ur5_values 
    
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
        rM = ruthModel()
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

        urdfDirectory = "/gym/envs/kelin/urdf/ur5_plus_RUTH.urdf" 
        robotPos = [0, 0, 0]
        robotScale = 1
        robot = pybullet.loadURDF(urdfDirectory,
                            robotPos,
                            pybullet.getQuaternionFromEuler([0, 0, 0]),
                            useFixedBase=0,
                            globalScaling=robotScale)

        #===== Check joint information from URDF file
        jointNum = pybullet.getNumJoints(robot)
        # print('\n\n', jointNum)
        for jt in range(jointNum):
            jointInfo = pybullet.getJointInfo(robot, jt)
            # print('\n\n',jointInfo)

        #===== Save UR5 joint and link information
        ur5JointNameToID = {}
        ur5LinkNameToID = {}
        ur5_joint_names = []
        ur5RevoluteID = []

        num_of_ur5_joints = 6

        for j in range(num_of_ur5_joints+2):
            info = pybullet.getJointInfo(robot, j)
            jointID = info[0]
            jointName = info[1].decode('UTF-8')
            jointType = info[2]
            ur5JointNameToID[jointName] = info[0]
            ur5LinkNameToID[info[12].decode('UTF-8')] = info[0]
            ur5RevoluteID.append(j)
            ur5_joint_names.append(jointName)



        # ===== Save RUTH joint and link information
        RUTH_jointNameToID = {} # Dictionary of RUTH joint and respective ID
        RUTH_linkNameToID = {} # Dictionary of RUTH link and respective ID
        RUTH_revoluteID = [] # array list of revolute joint ID?

        for j in range(num_of_ur5_joints+2, pybullet.getNumJoints(robot)): #range 8-23. RUTH joints/links
            info = pybullet.getJointInfo(robot, j)
            jointID = info[0]
            jointName = info[1].decode('UTF-8')
            jointType = info[2]
            RUTH_jointNameToID[jointName] = info[0]
            RUTH_linkNameToID[info[12].decode('UTF-8')] = info[0]
            RUTH_revoluteID.append(j)

        # print(RUTH_linkNameToID)

        #===== Disable Collisions between links
        for link in RUTH_linkNameToID:
            pybullet.setCollisionFilterGroupMask(robot, RUTH_linkNameToID[link], 1, 0)
        
        for link in ur5LinkNameToID:
            pybullet.setCollisionFilterGroupMask(robot, ur5LinkNameToID[link], 1, 0)
        # Disable collision between environments
        pybullet.setCollisionFilterGroupMask(robot, -1, 1, 0) 
        pybullet.setCollisionFilterGroupMask(floor, -1, 1, 0)

        # Set RUTH parallel links
        initialize_RUTH(robot, RUTH_linkNameToID, RUTH_jointNameToID)

        #======= Visualize contact points as spheres
        # contact_points = load_contact_pt(".\Contact_Points\Ruth\Ruth_Banana.npy", 1)
        # visualize_contact_pt(contact_points, robotScale)

        #=======Start simulation
        self.robot = robot
        self.RUTH_linkNameToID = RUTH_linkNameToID
        self.RUTH_jointNameToID = RUTH_jointNameToID
        self.ur5JointNameToID = ur5JointNameToID
        self.ur5_joint_names = ur5_joint_names 
        self.ur5LinkNameToID = ur5LinkNameToID

        self.pybulletDebug1 = pybulletDebug('motors', self.robot, self.ur5LinkNameToID)
        self._RUTH_init = [0. , 0. , 0. ]
        self._ur5_init = [0,np.pi/2,-np.pi/2,-np.pi/2,np.pi/2,0 ]
        
        self.version = version 

        # self.action_space = spaces.Box(np.array([-1.0,]*9), np.array([1.0,]*9))
        # self.observation_space = spaces.Box(np.array([0.0]*9), np.array([1.0]*9))
        self.action_space = spaces.Box(np.array([-1.0,]*9), np.array([1.0,]*9))
        self.observation_space = spaces.Box(np.array([0.0]*9), np.array([1.0]*9))
        self.goal_space = spaces.Box(np.array([0.0]*9), np.array([1.0]*9))
        
        self.max_delta_action = 0.2
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, flag=True, count=0):
        
        action = np.clip(action, -1, 1) 
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        action = action * self.max_delta_action 
        
        RUTH_motors, ur5_values = self.pybulletDebug1.return_robot_states() 
        RUTH_motors = action[:3] + RUTH_motors
        ur5_values = action[3:] + ur5_values
        # ur5_values = self.pybulletDebug1.return_robot_states() 
        # ur5_values = action + ur5_values
        target = np.concatenate((RUTH_motors, ur5_values))   
        motor_control_ur5(self.robot, self.ur5JointNameToID, self.ur5_joint_names, ur5_values)
        motor_control_ruth(self.robot, self.RUTH_jointNameToID, RUTH_motors)
        while(flag):
          pybullet.stepSimulation()
          real = self.get_obs()
          # real = np.asarray(real[3:])
          real = np.asarray(real[:9])
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
        if RUTH_motors[0]>=np.pi/2 or RUTH_motors[0]<=-1:
            r = -10
        if RUTH_motors[1]>=1 or RUTH_motors[1]<=-np.pi/2:
            r = -10
        if RUTH_motors[2]>=0.12 or RUTH_motors[2]<=-0.5:
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
        motor_control_ruth(self.robot, self.RUTH_jointNameToID, self._RUTH_init)
        target=np.array([0,np.pi/2,-np.pi/2,-np.pi/2,np.pi/2,0,0,0,0])
        self.reset_goal()
        while(flag):
          pybullet.stepSimulation()
          real = self.get_obs()
          real = np.asarray(real[:9])
          err = np.sum((real-target)*(real-target))
          count=count+1
          #print(err)
          if err < 1e-3 or count==100:
              flag = False
        pybullet.stepSimulation()
         
        state = self.get_obs() 
        return state 
    
    def reset_goal(self, goal=None): 
        if goal is None:
            x = random.uniform(0.3,0.5)
            y = random.uniform(-0.15,0.15)
            z = random.uniform(0.1,0.3)
            xyz = np.array([x,y,z])
            xyzd = random.rand(9)*0.1
            goal = np.array(xyz.tolist()*3) + xyzd 
        if goal.shape != (9,):
            print("the shape of goal should be (9,)")
        self.goal = goal 
        if self.version == "GUI":
            visualize_contact_pt(self.goal[:3], [1,0,0,1], 2)
            visualize_contact_pt(self.goal[3:6], [0,1,0,1], 2)
            visualize_contact_pt(self.goal[6:], [0,0,1,1], 2) 
        return self.goal 
    
    def get_obs(self):
        RUTH_motors, ur5_values = self.pybulletDebug1.return_robot_states()
        return np.concatenate((RUTH_motors, ur5_values, self.goal)).copy()
    
    # def get_obs(self):
    #     ur5_values = self.pybulletDebug1.return_robot_states()
    #     return np.concatenate((ur5_values, self.goal)).copy()
    
    def get_pos(self):
        #===========Get final fingertip position
        # getLinkState returns: 0. CoM coordinates, 1. CoM orientation
        fingertip_state = {}
        fingertip_pos = []
        fingertip_ori = [] #orientation of CenvoM of fingertip
        fingertip_links = ['Phal_1C','Phal_2C','Phal_3C']
        for link_name in fingertip_links:
            fingertip_state[link_name] = pybullet.getLinkState(self.robot, self.RUTH_linkNameToID[link_name])
            fingertip_pos.append(fingertip_state[link_name][0])
            fingertip_ori.append(fingertip_state[link_name][1])
        return np.array(fingertip_pos).ravel() 
    


#    def render(self, mode='human', **kwargs):
#        return self.env.render(mode, **kwargs)

    def render_frame(self, hw=15, imagesize=(300,300), camera_id=0, heatmap=None): 
        return None  

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

#======= Automate individual joint testing
def auto_joint_test( _ur5_joint_index=None, _ur5_init = 0, \
                    _ruth_motor_index = None, _ruth_init = 0, \
                    _ur5_motor_limit=[-np.pi, np.pi], _ruth_motor_limit = [[-1.0, np.pi/2], [-np.pi/2, 1.0], [-0.5, 0.12]]):
    total_step = 100
    _RUTH_motors = [0. , 0. , 0. ]
    _ur5_values = [0. , 0. , 0. , 0. , 0. , 0. ]
    # _ruth_motor_limit = [[-1.0, np.pi/2], [-np.pi/2, 1.0], [-0.5, 0.12]]
    # _ur5_motor_limit=[-np.pi, np.pi]

    #==== Start from lower limit if motor limit is given
    if _ur5_init is None and _ruth_init is None:
        if _ur5_motor_limit is not None:
            _ur5_init = _ur5_motor_limit[0]
        if _ruth_motor_limit is not None:
            _ruth_init = _ruth_motor_limit[0]

    #==== Test given individual joint
    if _ur5_joint_index is not None:
        _ur5_values[_ur5_joint_index] = _ur5_init
        for i in range(total_step):
            _ur5_values[_ur5_joint_index] = _ur5_values[_ur5_joint_index] + i*_ur5_motor_limit[1]/total_step

    if _ruth_motor_index is not None:
        _RUTH_motors[_ruth_motor_index] = _ruth_init
        for j in range(total_step):
            _RUTH_motors[_ruth_motor_index] = _RUTH_motors[_ruth_motor_index] + j*_ruth_motor_limit[_ruth_motor_index][1]/total_step


    return _RUTH_motors, _ur5_values


if __name__ == "__main__":
    import gym
    ro = gym.make("kelin-v0", version="GUI")
    sta = ro.reset() 
    ob = ro.get_obs()
    print(ob)
    for i in range(500):
        a = ro.action_space.sample() #* 0 + 0.1
        o, r, d, _ = ro.step(a)
        # print(o)
        if i % 20 == 0:
            ro.reset() 
        time.sleep(1./20.)
    sta = ro.reset() 
    ob = ro.get_obs()
    print(ob)
    pass 