# Set up UR5 and RUTH for simulation
# This file simply tests UR5 and RUTH movement control
# 2021-Oct-23 @Xian Zhang
#=========================READ ME==========================
# When calling main(mode, timer), first parameter takes 0 or 1 for control mode:
# (0) for manual motor control, (1) to start automated testing
# timer parameter is how long you want the simulation to run for.
# Edit automated testing in function auto_joint_test(), by specifying which joints to start auto testing
# Final UR5 and RUTH joint states, final fingertip position and orientation are saved in array output_motor_fingertip




import pybullet as p
import numpy as np
import time
import pybullet_data
from scipy.spatial.transform import Rotation as R
from ruth_grasping_kinematics import ruthModel
from pybullet_object_models import ycb_objects
import os
# import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd

class FingerAngles:
    def __init__(self, robot, linkNameToID):
        self.robot = robot
        self.linkNameToID = linkNameToID
        Phal_1A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_1A'])
        Phal_2A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_2A'])
        Phal_3A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_3A'])
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
        Phal_1A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_1A'])
        Phal_2A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_2A'])
        Phal_3A_info = p.getLinkState(self.robot, self.linkNameToID['Phal_3A'])
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
    return(cp_list)

#======= Visualize contact points as spheres
def visualize_contact_pt(point_pos, rgba, _robotScale):
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01 * _robotScale)
    visualShapeId = -1

    sphereA = p.createMultiBody(0.01, colSphereId, visualShapeId, point_pos)

    # Disable Collisions between links
    p.setCollisionFilterGroupMask(sphereA, -1, 1, 0)
    con2 = p.createConstraint(parentBodyUniqueId=sphereA,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=point_pos)

    p.changeVisualShape(sphereA, -1, rgbaColor=rgba)

#===== Join RUTH parallel links together, initialize motor positions to 0
def initialize_RUTH(_robot, _RUTH_linkNameToID, _RUTH_jointNameToID):
    link2CoM = [-0.0138504507471105, 0.00298913196612882, 0.0239892494327788]
    link2CoM2 = np.array([link2CoM[0], 0, link2CoM[2]])
    link2_Joint = -link2CoM2 + 0.07*link2CoM2/np.linalg.norm(link2CoM2)

    link4CoM = [0.0139032099348594, -0.00302765430432597, 0.0240810659954725]
    link4CoM2 = np.array([link4CoM[0], 0, link4CoM[2]])
    link4_Joint = -link4CoM2 + 0.07*link4CoM2/np.linalg.norm(link4CoM2)

    link2_info = p.getLinkState(_robot, _RUTH_linkNameToID['Link_2'])
    link4_info = p.getLinkState(_robot, _RUTH_linkNameToID['Link_4'])
    CoM_diff = np.array(link4_info[0]) - np.array(link2_info[0])
    link2_Joint = link2_Joint + np.array([0, CoM_diff[2], 0])

    con1 = p.createConstraint(parentBodyUniqueId=_robot,
                        parentLinkIndex= _RUTH_linkNameToID['Link_2'],
                        childBodyUniqueId=_robot,
                        childLinkIndex= _RUTH_linkNameToID['Link_4'],
                        jointType=p.JOINT_POINT2POINT,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[link2_Joint[0], link2_Joint[1], link2_Joint[2]],
                        childFramePosition=[link4_Joint[0], link4_Joint[1], link4_Joint[2]])
    
    initPos1 = 1*np.pi/100000
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, -initPos1, force=1000)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, initPos1, force=1000)

#===== Joint motor control on UR5
def motor_control_ur5(_robot, _ur5JointNameToID, _ur5_joint_name, _ur5_values):
    for index, ur5_control in enumerate(_ur5_values):   #ur5_control = ur5_values[index]
        p.setJointMotorControl2(_robot, _ur5JointNameToID[_ur5_joint_name[index+1]], p.POSITION_CONTROL, _ur5_values[index], force=1000)

#===== Joint motor control on RUTH
def motor_control_ruth(_robot, _RUTH_jointNameToID, _ruth_base_motors):
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, _ruth_base_motors[0], force=1000)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, _ruth_base_motors[1], force=100)  

    # Calculate relative individual finger joint movements
    finger_angles = [0, 0, 0]
    finger_pos_plus = 0.12745044
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_1A'], p.POSITION_CONTROL, -finger_angles[0], force=1000)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_2A'], p.POSITION_CONTROL, finger_angles[1], force=100)  
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_3A'], p.POSITION_CONTROL, finger_angles[2], force=100)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_1B'], p.POSITION_CONTROL, -0.55-_ruth_base_motors[2]-finger_pos_plus, force=1000)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_2B'], p.POSITION_CONTROL, -0.55-_ruth_base_motors[2]-finger_pos_plus, force=100)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_3B'], p.POSITION_CONTROL, 0.65+_ruth_base_motors[2]+finger_pos_plus, force=100) 
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_1C'], p.POSITION_CONTROL, 1-_ruth_base_motors[2]-finger_pos_plus, force=100)   
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_2C'], p.POSITION_CONTROL, 1-_ruth_base_motors[2]-finger_pos_plus, force=1000)
    p.setJointMotorControl2(_robot, _RUTH_jointNameToID['Joint_Phal_3C'], p.POSITION_CONTROL, -1+_ruth_base_motors[2]+finger_pos_plus, force=100)

#======= Config parameter panel
class pybulletDebug:
    def __init__(self, control_type, robot, ur5LinkNameToID):
        #Camera paramers to be able to yaw pitch and zoom the camera (Focus remains on the robot) 
        self.cyaw=90
        self.cpitch=-7
        self.cdist=0.66
        time.sleep(0.5)
        self.init_state = p.getLinkState(robot, ur5LinkNameToID['ee_link'])
        self.init_pos = np.array(self.init_state[0])
        self.init_ori = np.array(p.getEulerFromQuaternion(self.init_state[1]))

       
        if control_type == 'motors':
            self.control_type = 'motors'
            self.U1Id = p.addUserDebugParameter("UR5 Shoulder Pan" , -3.14 , 3.14 , 0.)
            self.U2Id = p.addUserDebugParameter("UR5 Shoulder Lift" , -3.14 , 3.14 , 0.)
            self.U3Id = p.addUserDebugParameter("UR5 Elbow" , -3.14 , 3.14 , 0.)
            self.U4Id = p.addUserDebugParameter("UR5 Wrist 1" , -3.14 , 3.14 , 0.)
            self.U5Id = p.addUserDebugParameter("UR5 Wrist 2" , -3.14 , 3.14 , 0.)
            self.U6Id = p.addUserDebugParameter("UR5 Wrist 3" , -3.14 , 3.14 , 0.)
    
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
        self.Rm1Id = p.addUserDebugParameter("RUTH Motor 1" , -1.0 , 1.57 , 0.) 
        self.Rm2Id = p.addUserDebugParameter("RUTH Motor 2" , -1.57 , 1.0 , 0.) 
        self.RfId = p.addUserDebugParameter("RUTH fingers" , -0.5 , 0.12 , 0.)

    
    def return_robot_states(self):
        motor_positions = np.array([p.readUserDebugParameter(self.Rm1Id),p.readUserDebugParameter(self.Rm2Id), p.readUserDebugParameter(self.RfId)]) 
        # RUTH motor 1 [-1.57 , 3.14], motor 2 [-3.14 , 1.57],  RUTH finger bend [-0.5, 0.5]
        ur5_values = np.array([p.readUserDebugParameter(self.U1Id),p.readUserDebugParameter(self.U2Id),p.readUserDebugParameter(self.U3Id),p.readUserDebugParameter(self.U4Id),p.readUserDebugParameter(self.U5Id),p.readUserDebugParameter(self.U6Id)])
       
        return motor_positions, ur5_values

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


#==================================MAIN()=================================
def main(mode, timer):
    rM = ruthModel()
    #========= Initialize environment
    # physicsClient = p.connect(p.GUI)
    physicsClient = p.connect(p.DIRECT) #non-graphical version

    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-9.81)
    p.setRealTimeSimulation(0) #0: disable

    #===== Initialize plane and import robot
    FixedBase = False #if fixed no plane is imported
    if (FixedBase == False):
        floor = p.loadURDF("plane.urdf")

    urdfDirectory = "/gym/envs/kelin/urdf/ur5_plus_RUTH.urdf" 
    robotPos = [0, 0, 0]
    robotScale = 1
    robot = p.loadURDF(urdfDirectory,
                        robotPos,
                        p.getQuaternionFromEuler([0, 0, 0]),
                        useFixedBase=0,
                        globalScaling=robotScale)

    #===== Check joint information from URDF file
    jointNum = p.getNumJoints(robot)
    # print('\n\n', jointNum)
    for jt in range(jointNum):
        jointInfo = p.getJointInfo(robot, jt)
        # print('\n\n',jointInfo)

    #===== Save UR5 joint and link information
    ur5JointNameToID = {}
    ur5LinkNameToID = {}
    ur5_joint_names = []
    ur5RevoluteID = []

    num_of_ur5_joints = 6

    for j in range(num_of_ur5_joints+2):
        info = p.getJointInfo(robot, j)
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

    for j in range(num_of_ur5_joints+2,p.getNumJoints(robot)): #range 8-23. RUTH joints/links
        info = p.getJointInfo(robot, j)
        jointID = info[0]
        jointName = info[1].decode('UTF-8')
        jointType = info[2]
        RUTH_jointNameToID[jointName] = info[0]
        RUTH_linkNameToID[info[12].decode('UTF-8')] = info[0]
        RUTH_revoluteID.append(j)

    # print(RUTH_linkNameToID)

    #===== Disable Collisions between links
    for link in RUTH_linkNameToID:
        p.setCollisionFilterGroupMask(robot, RUTH_linkNameToID[link], 1, 0)
    
    for link in ur5LinkNameToID:
        p.setCollisionFilterGroupMask(robot, ur5LinkNameToID[link], 1, 0)
    # Disable collision between environments
    p.setCollisionFilterGroupMask(robot, -1, 1, 0) 
    p.setCollisionFilterGroupMask(floor, -1, 1, 0)

    # Set RUTH parallel links
    initialize_RUTH(robot, RUTH_linkNameToID, RUTH_jointNameToID)


    #======= Visualize contact points as spheres
    # contact_points = load_contact_pt(".\Contact_Points\Ruth\Ruth_Banana.npy", 1)
    # visualize_contact_pt(contact_points, robotScale)

    #=======Start simulation
    pybulletDebug1 = pybulletDebug('motors',robot,ur5LinkNameToID)
    _RUTH_init = [0. , 0. , 0. ]
    _ur5_init = [0. , 0. , 0. , 0. , 0. , 0. ]
    # timer = 200
    # mode = 1

    for i in range (timer):    
        if mode == 0: #User input motor values from GUI
            RUTH_motors, ur5_values = pybulletDebug1.return_robot_states() 
            motor_control_ur5(robot, ur5JointNameToID,ur5_joint_names, ur5_values)
            motor_control_ruth(robot, RUTH_jointNameToID, RUTH_motors)
            p.stepSimulation()
            time.sleep(1./20.)

        elif mode ==1: #Automate motor controls
            RUTH_motors, ur5_values = auto_joint_test(_ur5_joint_index= 0, _ur5_init = -1.57, _ruth_motor_index = None, _ruth_init = 0)

            if i<timer/2 :
                #indicate which joints to test by indicating '_ur5_joint_index' and '_ruth_motor_index'
                motor_control_ur5(robot, ur5JointNameToID,ur5_joint_names, ur5_values)
                motor_control_ruth(robot, RUTH_jointNameToID, RUTH_motors)
                p.stepSimulation()
                time.sleep(1./20.)

            #return to init state
            else:
                motor_control_ur5(robot, ur5JointNameToID,ur5_joint_names, _ur5_init)
                motor_control_ruth(robot, RUTH_jointNameToID, _RUTH_init)
                p.stepSimulation()
                time.sleep(1./20.)


        


        #===========Get final fingertip position
        # getLinkState returns: 0. CoM coordinates, 1. CoM orientation
        fingertip_state = {}
        fingertip_pos = []
        fingertip_ori = [] #orientation of CoM of fingertip
        fingertip_links = ['Phal_1C','Phal_2C','Phal_3C']
        for link_name in fingertip_links:
            fingertip_state[link_name] = p.getLinkState(robot, RUTH_linkNameToID[link_name])
            fingertip_pos.append(fingertip_state[link_name][0])
            fingertip_ori.append(fingertip_state[link_name][1])

        visualize_contact_pt(fingertip_pos[0], [1,0,0,1], robotScale)
        visualize_contact_pt(fingertip_pos[1], [0,1,0,1],robotScale)
        visualize_contact_pt(fingertip_pos[2], [0,0,1,1],robotScale)
        


        output_motor_fingertip = [RUTH_motors , ur5_values, fingertip_pos, fingertip_ori]    
        
        # print(output_motor_fingertip)
        print(fingertip_pos[0])
        print(fingertip_pos[1])
        print(fingertip_pos[2])
        
 
    

if __name__ == "__main__":
    main(0, 1000)