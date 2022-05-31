import pybullet as p
import numpy as np
import time
import pybullet_data
from pybullet_debuger_ur5_plus_ruth import pybulletDebug  
import click
from scipy.spatial.transform import Rotation as R
from ruth_grasping_kinematics_0 import ruthModel


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

rM = ruthModel()

@click.command()
@click.option('--control_type', type=str, default='xyz', help='--xyz for control via end-effector pose, --motors for control via motor positions')

def main(control_type):

    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-9.8)


    urdfDirectory = "../urdf/ur5_plus_RUTH.urdf"
    #urdf_root = pybullet_data.getDataPath()

    cubeStartPos = [0,0,0]
    FixedBase = False #if fixed no plane is imported
    if (FixedBase == False):
        p.loadURDF("plane.urdf")

    robotPos = [0, 0, 0]
    robotScale = 1
    robot = p.loadURDF(urdfDirectory,
                       robotPos,
                       p.getQuaternionFromEuler([0, 0, 0]),
                       useFixedBase=0,
                       globalScaling=robotScale)

    duck = p.loadURDF("duck_vhacd.urdf", [0.5, 0, 0])

    ur5JointNameToID = {}
    ur5LinkNameToID = {}
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

    jointNameToID = {}
    linkNameToID = {}
    revoluteID = []
    for j in range(num_of_ur5_joints+2,p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        jointID = info[0]
        jointName = info[1].decode('UTF-8')
        jointType = info[2]
        jointNameToID[jointName] = info[0]
        linkNameToID[info[12].decode('UTF-8')] = info[0]
        revoluteID.append(j)

    pybulletDebug1 = pybulletDebug(control_type, robot, ur5LinkNameToID)

    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01 * robotScale)
    visualShapeId = -1

    #ballposA = [0.1483995492528895, -0.04204383314160415, 0.20278895201841407+0.2]
    #sphereA = p.createMultiBody(0.01, colSphereId, visualShapeId, ballposA)
    #sphereB = p.createMultiBody(0.01, colSphereId, visualShapeId, ballposA)


#=======Disable Collisions between links
    for link in linkNameToID:
        p.setCollisionFilterGroupMask(robot, linkNameToID[link], 1, 0)
    p.setCollisionFilterGroupMask(robot, -1, 1, 0) 
    #p.setCollisionFilterGroupMask(sphereA, -1, 1, 0)
    #p.setCollisionFilterGroupMask(sphereB, -1, 1, 0)

    link2CoM = [-0.0138504507471105, 0.00298913196612882, 0.0239892494327788]
    link2CoM2 = np.array([link2CoM[0], 0, link2CoM[2]])
    link2_Joint = -link2CoM2 + 0.07*link2CoM2/np.linalg.norm(link2CoM2)

    link4CoM = [0.0139032099348594, -0.00302765430432597, 0.0240810659954725]
    link4CoM2 = np.array([link4CoM[0], 0, link4CoM[2]])
    link4_Joint = -link4CoM2 + 0.07*link4CoM2/np.linalg.norm(link4CoM2)

    link2_info = p.getLinkState(robot, linkNameToID['Link_2'])
    link4_info = p.getLinkState(robot, linkNameToID['Link_4'])
    CoM_diff = np.array(link4_info[0]) - np.array(link2_info[0])
    link2_Joint = link2_Joint + np.array([0, CoM_diff[2], 0])

#======= RUTH 5-linkage system is not open-ended robot, its control system is considered a parallel robot. 
#======= This is not allowed by the system, therefore create link 2 and 4 seperatedly, 
#======= Then make them meet at the same point of constraint to create the closed loop control.
    con1 = p.createConstraint(parentBodyUniqueId=robot,
                       parentLinkIndex=linkNameToID['Link_2'],
                       childBodyUniqueId=robot,
                       childLinkIndex=linkNameToID['Link_4'],
                       jointType=p.JOINT_POINT2POINT,
                       jointAxis=[0, 0, 0],
                       parentFramePosition=[link2_Joint[0], link2_Joint[1], link2_Joint[2]],
                       childFramePosition=[link4_Joint[0], link4_Joint[1], link4_Joint[2]])

    #p.createConstraint(parentBodyUniqueId=robot,
    #                   parentLinkIndex=linkNameToID['Link_2'],
    #                   childBodyUniqueId=sphereA,
    #                   childLinkIndex=-1,
    #                   jointType=p.JOINT_POINT2POINT,
    #                   jointAxis=[0, 0, 0],
    #                   parentFramePosition=[link2_Joint[0], link2_Joint[1], link2_Joint[2]],
    #                   childFramePosition=[0, 0, 0])

    #p.createConstraint(parentBodyUniqueId=robot,
    #                   parentLinkIndex=linkNameToID['Link_4'],
    #                   childBodyUniqueId=sphereB,
    #                   childLinkIndex=-1,
    #                   jointType=p.JOINT_POINT2POINT,
    #                   jointAxis=[0, 0, 0],
    #                   parentFramePosition=[link4_Joint[0], link4_Joint[1], link4_Joint[2]],
    #                   childFramePosition=[0, 0, 0])

    #p.createConstraint(parentBodyUniqueId=sphereA,
    #                   parentLinkIndex=-1,
    #                   childBodyUniqueId=sphereB,
    #                   childLinkIndex=-1,
    #                   jointType=p.JOINT_FIXED,
    #                   jointAxis=[0, 0, 0],
    #                   parentFramePosition=[0, 0, 0],
    #                   childFramePosition=[0, 0, 0])

    p.stepSimulation()
    time.sleep(1./240.)

    #p.createConstraint(parentBodyUniqueId=robot,
    #                   parentLinkIndex=-1,
    #                   childBodyUniqueId=ur5,
    #                   childLinkIndex=ur5LinkNameToID['wrist_3_link'],
    #                   jointType=p.JOINT_FIXED,
    #                   jointAxis=[0, 0, 0],
    #                   parentFramePosition=[0, 0, 0],
    #                   childFramePosition=[0, .15, 0],
    #                   childFrameOrientation=p.getQuaternionFromEuler([-1.57,0,0]))

    p.setJointMotorControl2(robot, jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, 0.0, force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, 0.0, force=1000)


    contact_point1 = np.array([0.65, -0.2, 0.35])
    contact_point2 = np.array([0.625, -0.175, 0.35])
    contact_point3 = np.array([0.625, -0.225, 0.35])
    # contact_point1 = np.array([0.28, 0.008, 0.025])  # Test points for banana
    # contact_point2 = np.array([0.31, -0.008, 0.025])
    # contact_point3 = np.array([0.34, 0.008, 0.025])
    CentrePoint, finger_bend = rM.compute_ruth_pose(contact_point1*1000, contact_point2*1000, contact_point3*1000)
    CentrePoint = CentrePoint/1000

    R_x_ax = (contact_point2-contact_point3)/np.linalg.norm(contact_point2-contact_point3)
    R_y_ax = np.cross(R_x_ax,(contact_point1-contact_point3)/np.linalg.norm(contact_point1-contact_point3))/np.linalg.norm(np.cross(R_x_ax,(contact_point1-contact_point3)/np.linalg.norm(contact_point1-contact_point3)))
    R_z_ax = np.cross(R_x_ax, R_y_ax)
    R_CPs = np.column_stack((R_x_ax, R_y_ax, R_z_ax))
    if R_y_ax[2]<0: # If 
        R_CPs = np.matmul(np.column_stack(([-1,0,0], [0,-1,0], [0,0,1])),R_CPs)
    R_CPs2 = np.matmul(np.column_stack(([0,0,-1], [0,1,0], [1,0,0])),R_CPs)
    r = R.from_matrix(R_CPs)
    CPs_Eulers = p.getEulerFromQuaternion(r.as_quat())

    sphereA = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point1)
    sphereB = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point2)
    sphereC = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point3)
    # Disable Collisions between links
    p.setCollisionFilterGroupMask(sphereA, -1, 1, 0)
    p.setCollisionFilterGroupMask(sphereB, -1, 1, 0)
    p.setCollisionFilterGroupMask(sphereC, -1, 1, 0)
    con2 = p.createConstraint(parentBodyUniqueId=sphereA,
               parentLinkIndex=-1,
               childBodyUniqueId=-1,
               childLinkIndex=-1,
               jointType=p.JOINT_FIXED,
               jointAxis=[0, 0, 0],
               parentFramePosition=[0, 0, 0],
               childFramePosition=contact_point1)
    con3 = p.createConstraint(parentBodyUniqueId=sphereB,
               parentLinkIndex=-1,
               childBodyUniqueId=-1,
               childLinkIndex=-1,
               jointType=p.JOINT_FIXED,
               jointAxis=[0, 0, 0],
               parentFramePosition=[0, 0, 0],
               childFramePosition=contact_point2)
    con4 = p.createConstraint(parentBodyUniqueId=sphereC,
               parentLinkIndex=-1,
               childBodyUniqueId=-1,
               childLinkIndex=-1,
               jointType=p.JOINT_FIXED,
               jointAxis=[0, 0, 0],
               parentFramePosition=[0, 0, 0],
               childFramePosition=contact_point3)


    for i in range (1000):
        p.stepSimulation()
        motor_positions, finger_position, ur5_values = pybulletDebug1.cam_and_robotstates(robot, linkNameToID['ruth_base'])
#        print(p.getEulerFromQuaternion(p.getLinkState(robot, linkNameToID['Link_1'])[5]))
#        print(p.getEulerFromQuaternion(p.getLinkState(robot, linkNameToID['ruth_base'])[5]))
#        print('\n')
#        h/g
        time.sleep(1./240.)




    p.setRealTimeSimulation(0)
    targetPos1 = 0.0
    targetPos2 = 0.0
    FA = FingerAngles(robot, linkNameToID) # Finger angles class


    for i in range(1000):
        motor_positions, finger_position, ur5_values = pybulletDebug1.cam_and_robotstates(robot, linkNameToID['ruth_base']) # Camera control
    #    targetPos1 = -i*np.pi/100000
    #    targetPos2 = i*np.pi/100000
        targetPos1 = motor_positions[0]
        targetPos2 = motor_positions[1]
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, targetPos1, force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, targetPos2, force=1000)
        finger_angles = FA.calc_finger_angles()
        p.stepSimulation()


    i=0
    while(True):
        motor_positions, finger_position, ur5_values = pybulletDebug1.cam_and_robotstates(robot, ur5LinkNameToID['ee_link'])


        if control_type=='motors': # motor control
            ur5_joints = ur5_values
#            ee_state = p.getLinkState(robot, ur5LinkNameToID['ee_link'])
#            ee_pos, ee_ori = ee_state[4], ee_state[5]
#            robot_joints = p.calculateInverseKinematics(robot, ur5LinkNameToID['ee_link'], ee_pos, ee_ori)
#            ur5_joints = robot_joints[:6]
        else: # end-effector pose control
            ur5_values[3] = CPs_Eulers[0] # + ur5_values[3]
            ur5_values[4] = CPs_Eulers[2] # + ur5_values[4]       
            ur5_values[5] = CPs_Eulers[1] # + ur5_values[5]
            ee_pos, ee_ori = ur5_values[:3], p.getQuaternionFromEuler(ur5_values[3:])
            ee_ori = R.from_matrix(R_CPs2).as_quat()
            ee_pos = CentrePoint

            robot_joints = p.calculateInverseKinematics(robot, linkNameToID['tcp'], ee_pos, ee_ori)
            ur5_joints = robot_joints[:6]
            ee_pos2 = p.getLinkState(robot, ur5LinkNameToID['wrist_3_link'])
#            print(ee_pos)            
#            print(ee_pos2[4])

        if i<1000:
            i+=1
            finger_pos_plus = 0
        else:
            i+=1
            finger_pos_plus = finger_bend


        targetPos1 = motor_positions[0]
        targetPos2 = motor_positions[1]
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, targetPos1, force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, targetPos2, force=100)  
        finger_angles = FA.calc_finger_angles()
        finger_angles = [0, 0, 0]
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_1A'], p.POSITION_CONTROL, -finger_angles[0], force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_2A'], p.POSITION_CONTROL, finger_angles[1], force=100)  
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_3A'], p.POSITION_CONTROL, finger_angles[2], force=100)


        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_1B'], p.POSITION_CONTROL, -0.55-finger_position-finger_pos_plus, force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_1C'], p.POSITION_CONTROL, 1-finger_position-finger_pos_plus, force=100)  
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_2B'], p.POSITION_CONTROL, -0.55-finger_position-finger_pos_plus, force=100)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_2C'], p.POSITION_CONTROL, 1-finger_position-finger_pos_plus, force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_3B'], p.POSITION_CONTROL, 0.65+finger_position+finger_pos_plus, force=100)  
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_3C'], p.POSITION_CONTROL, -1+finger_position+finger_pos_plus, force=100)

        p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_pan_joint'], p.POSITION_CONTROL, ur5_joints[0], force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_lift_joint'], p.POSITION_CONTROL, ur5_joints[1], force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['elbow_joint'], p.POSITION_CONTROL, ur5_joints[2], force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['wrist_1_joint'], p.POSITION_CONTROL, ur5_joints[3], force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['wrist_2_joint'], p.POSITION_CONTROL, ur5_joints[4], force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['wrist_3_joint'], p.POSITION_CONTROL, ur5_joints[5], force=1000)

        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()


if __name__ == "__main__":
    main()



