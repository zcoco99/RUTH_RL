import pybullet as p
import numpy as np
import time
import pybullet_data
from pybullet_debuger_ur5_plus_ruth import pybulletDebug  
import click
from scipy.spatial.transform import Rotation as R
from ruth_grasping_kinematics import ruthModel
from pybullet_object_models import ycb_objects
import os
import open3d as o3d
import matplotlib.pyplot as plt
import sys


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


    urdfDirectory = "./RUTH_Gripper/urdf/ur5_plus_RUTH.urdf"
    #urdf_root = pybullet_data.getDataPath()

    cubeStartPos = [0,0,0]
    FixedBase = False #if fixed no plane is imported
    if (FixedBase == False):
        floor = p.loadURDF("plane.urdf")

    robotPos = [0, 0, 0.2]
    robotScale = 1
    robot = p.loadURDF(urdfDirectory,
                       robotPos,
                       p.getQuaternionFromEuler([0, 0, 0]),
                       useFixedBase=1,
                       globalScaling=robotScale)

#    duck = p.loadURDF("duck_vhacd.urdf", [0.5, 0, 0])

    flags = p.URDF_USE_INERTIA_FROM_FILE
    obj_pos = [0.3, 0, 0.01]
    obj_ori = p.getQuaternionFromEuler([0,0,0])
    obj = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbBanana', "model.urdf"), obj_pos, baseOrientation=obj_ori, useFixedBase=0, flags=flags)
#    p.setCollisionFilterGroupMask(obj, -1, 1, 0)  # Disable object collisions for now

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

#    pybulletDebug1 = pybulletDebug(control_type, robot, ur5LinkNameToID)

    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01 * robotScale)
    visualShapeId = -1

    #ballposA = [0.1483995492528895, -0.04204383314160415, 0.20278895201841407+0.2]
    #sphereA = p.createMultiBody(0.01, colSphereId, visualShapeId, ballposA)
    #sphereB = p.createMultiBody(0.01, colSphereId, visualShapeId, ballposA)


    # Disable Collisions between links
    for link in linkNameToID:
        p.setCollisionFilterGroupMask(robot, linkNameToID[link], 1, 0)
    for link in ur5LinkNameToID:
        p.setCollisionFilterGroupMask(robot, ur5LinkNameToID[link], 1, 0)
    p.setCollisionFilterGroupMask(robot, -1, 1, 0) 
    p.setCollisionFilterGroupMask(floor, -1, 1, 0) 
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

#    ##########

#    # Set contact points and compute gripper position 
##    contact_point1 = np.array([0.65, -0.2, 0.4])
##    contact_point2 = np.array([0.65, -0.175, 0.425])
##    contact_point3 = np.array([0.65, -0.225, 0.425])
#    contact_point1 = np.array([0.65, -0.2, 0.38])
#    contact_point2 = np.array([0.625, -0.175, 0.35])
#    contact_point3 = np.array([0.625, -0.225, 0.33])
#    CentrePoint, finger_bend, R_CPs_test, R_CPs_test_T = rM.compute_ruth_pose(contact_point1*1000, contact_point2*1000, contact_point3*1000)
#    CentrePoint = CentrePoint/1000 # Compute RUTH position for inverse kinematics
#    

#    R_x_ax = (contact_point2-contact_point3)/np.linalg.norm(contact_point2-contact_point3)  # x-axis across 'base' points 
#    R_y_ax = np.cross(R_x_ax,(contact_point1-contact_point3)/np.linalg.norm(contact_point1-contact_point3))/np.linalg.norm(np.cross(R_x_ax,(contact_point1-contact_point3)/np.linalg.norm(contact_point1-contact_point3)))  # y-axis points normal to plane formed by 3 points
#    R_z_ax = np.cross(R_x_ax, R_y_ax)
#    R_CPs = np.column_stack((R_x_ax, R_y_ax, R_z_ax))

#    approach_distance = 0.1  # distance from which the robot approaches the grasp
#    approach_steps = 500
#    approach_step_size = approach_distance/approach_steps
#    CentrePoint_init = CentrePoint - R_y_ax*approach_distance  # approach normal to contact point plane 

#    if R_y_ax[2]<0: # If 
#        R_CPs = np.matmul(np.column_stack(([-1,0,0], [0,-1,0], [0,0,1])),R_CPs)

#    # R_CPs2 = np.matmul(np.column_stack(([0,0,-1], [0,1,0], [1,0,0])),R_CPs)
#    # R_CPs_to_G = np.matmul(R_CPs2,np.matrix.transpose(R_CPs_test))
#    R_CPs_to_G = np.column_stack(([-1.,0.,0.], [0.,-1.,0.], [0.,0.,1.]))

##    R_CPs2 = np.matmul(R_CPs_to_G, R_CPs_test)
##    R_CPs2 = np.column_stack(([0.,0.,1.], [-1.,0.,0.], [0.,-1.,0.]))
#    R_CPs2 = np.column_stack((R_CPs_test[:,0], -R_CPs_test[:,2], R_CPs_test[:,1]))

##    print(R_CPs_test_T)
##    print(np.matmul(np.matmul(R_CPs2,np.matrix.transpose(R_CPs_test)), R_CPs_test))
##    print(R_CPs2)
#    r = R.from_matrix(R_CPs)
#    CPs_Eulers = p.getEulerFromQuaternion(r.as_quat())


    ######################
    # Visualise contact points
        
#    sphereA = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point1)
#    sphereB = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point2)
#    sphereC = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point3)
#    # Disable Collisions between links
#    p.setCollisionFilterGroupMask(sphereA, -1, 1, 0)
#    p.setCollisionFilterGroupMask(sphereB, -1, 1, 0)
#    p.setCollisionFilterGroupMask(sphereC, -1, 1, 0)
#    con2 = p.createConstraint(parentBodyUniqueId=sphereA,
#               parentLinkIndex=-1,
#               childBodyUniqueId=-1,
#               childLinkIndex=-1,
#               jointType=p.JOINT_FIXED,
#               jointAxis=[0, 0, 0],
#               parentFramePosition=[0, 0, 0],
#               childFramePosition=contact_point1)
#    con3 = p.createConstraint(parentBodyUniqueId=sphereB,
#               parentLinkIndex=-1,
#               childBodyUniqueId=-1,
#               childLinkIndex=-1,
#               jointType=p.JOINT_FIXED,
#               jointAxis=[0, 0, 0],
#               parentFramePosition=[0, 0, 0],
#               childFramePosition=contact_point2)
#    con4 = p.createConstraint(parentBodyUniqueId=sphereC,
#               parentLinkIndex=-1,
#               childBodyUniqueId=-1,
#               childLinkIndex=-1,
#               jointType=p.JOINT_FIXED,
#               jointAxis=[0, 0, 0],
#               parentFramePosition=[0, 0, 0],
#               childFramePosition=contact_point3)

    ######################

#======= Depth camera set up and configurations

    fov_rad = np.pi*45.0/180.
    image_height = 1000
    image_width = image_height
    image_cx = image_height/2
    image_cy = image_width/2
    f = image_height/(2*np.tan(fov_rad/2))
    far = 3.1
    near = 0.1
    aspect_ratio = 1.

    viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[1, 0, 1],
    cameraTargetPosition=[0.3, 0, 0],
    cameraUpVector=[0, 0, 1])
    projectionMatrix = p.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=near,
    farVal=far)
#    print(2.0*np.arctan( 1.0/np.asarray(projectionMatrix).reshape((4,4))[1][1] ) * 180.0 / np.pi)


    top = np.tan(fov_rad/2)*near
    bottom = -top
    right = top*aspect_ratio
    left = -top*aspect_ratio
    proj_mat = np.array([[2*near/(right-left), 0, (right+left)/(right-left), 0], [0, 2*near/(top-bottom), (top+bottom)/(top-bottom), 0], [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)], [0, 0, -1, 0]])
#    print(np.asarray(projectionMatrix).reshape((4,4)))
#    print(np.transpose(proj_mat))
    
    for i in range (200):
        p.stepSimulation()
#        motor_positions, finger_position, ur5_values = pybulletDebug1.cam_and_robotstates(robot, linkNameToID['ruth_base'])
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
#        motor_positions, finger_position, ur5_values = pybulletDebug1.cam_and_robotstates(robot, linkNameToID['ruth_base']) # Camera control
        targetPos1 = -i*np.pi/100000
        targetPos2 = i*np.pi/100000
#        targetPos1 = motor_positions[0]
#        targetPos2 = motor_positions[1]

#=================RUTH Initialize positions
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, targetPos1, force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, targetPos2, force=1000)
        finger_angles = FA.calc_finger_angles()
#================UR5 position intialize position
        p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_pan_joint'], p.POSITION_CONTROL, 0, force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_lift_joint'], p.POSITION_CONTROL, np.pi/2, force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['elbow_joint'], p.POSITION_CONTROL, 0, force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['wrist_1_joint'], p.POSITION_CONTROL, 0, force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['wrist_2_joint'], p.POSITION_CONTROL, 0, force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['wrist_3_joint'], p.POSITION_CONTROL, 0, force=1000)
        p.stepSimulation()

    i=0
    approach_step_count = 0
    ur5_values = [0]*6
    motor_positions = [0, 0]
    finger_position = 0

    visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 1, 1, 1], radius=0.005)
    collisionShapeId = -1

    while(True):
#        motor_positions, finger_position, ur5_values = pybulletDebug1.cam_and_robotstates(robot, ur5LinkNameToID['ee_link'])
        
        image_width
        flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=image_width, 
            height=image_height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix, flags=flags)
        depthImgb = far * near / (far - (far-near)*depthImg)
        rgbImg2 = o3d.geometry.Image((np.array(rgbImg)[:,:,:3]).astype(np.uint8))
        depthImg2 = o3d.geometry.Image(depthImgb)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgbImg2, depthImg2)

#        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
#    rgbd_image,
#    o3d.camera.PinholeCameraIntrinsic(
#        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(image_width, image_height, f, f, image_width/2, image_height/2))

        if i==0:
            pcd_points = np.asarray(pcd.points)
            ii = 0
            pcd2 = np.empty((0,3))
            rgbd_acc = np.zeros((image_width, image_height))
            print(np.asarray(rgbImg).shape)
            rgb_test  = np.zeros((image_width, image_height,3))
            for j in range(image_width):
                for k in range(image_height):
                    rgbd_acc[j][k] = ii/image_width**2
                    ii += 1
                    if np.asarray(segImg[j][k]) == obj:
                        rgb_test[j][k][:] = 255
                        pcd2 = np.vstack((pcd2, pcd_points[ii,:]))  # Collect point cloud oordinates which correspond to object
#            print(max(np.asarray(pcd2)[:,0]*1000)-min(np.asarray(pcd2)[:,0]*1000))
#            pcd2_rot = np.matmul(np.array([[-1,0,0],[0,-1,0],[0,0,1]]),np.copy(-pcd2))
            pcd3 = o3d.geometry.PointCloud()
            pcd3.points = o3d.utility.Vector3dVector(pcd2)
            pcd4 = np.vstack((np.transpose(-pcd2*1000), np.ones(len(pcd2))))  # Homogenous coordinates and scale up by depth scale (1000)
            view_matrix2 = np.transpose(np.asarray(viewMatrix).reshape((4,4)))
            pcd5 = np.matmul(np.linalg.inv(view_matrix2), pcd4)  # convert from point cloud coordinates to world coordinates
            pcd5[1, :] = -pcd5[1, :]
#            print('Mv*P_C')            
#            print(np.matmul(np.linalg.inv(view_matrix2), np.matmul(view_matrix2, np.array([0.5, 0, 0, 1]))))
#            print(pcd2[0])
#            print(pcd4[:, 0])
#            print(pcd5)
#            pcd5_avx = np.sum(pcd5[0, :])/len(pcd5[0])
#            pcd5_avy = np.sum(pcd5[1, :])/len(pcd5[1])
#            pcd5_avz = np.sum(pcd5[2, :])/len(pcd5[2])
#            print(pcd5_avx, pcd5_avy, pcd5_avz)

            print(pcd2.shape)
            # o3d.visualization.draw_geometries([pcd3])  # Plot point cloud in Open3D
            
            sample_rate = int(np.floor(pcd2.shape[0]/2048))
            pcd6 =  o3d.geometry.PointCloud.uniform_down_sample(pcd3, sample_rate)
            print(np.asarray(pcd6.points).shape, sample_rate)
            # o3d.visualization.draw_geometries([pcd6])  # Plot point cloud in Open3D

            sample_rate = 2048/np.asarray(pcd6.points).shape[0]
            pcd6 = pcd6.random_down_sample(sample_rate)
            print(np.asarray(pcd6.points).shape, sample_rate)
            # o3d.visualization.draw_geometries([pcd6])  # Plot point cloud in Open3D


#            plt.subplot(1, 2, 1)
#            plt.title('Redwood grayscale image')
#            plt.imshow(rgb_test)
#            plt.subplot(1, 2, 2)
#            plt.title('Redwood depth image')
#            plt.imshow(rgbd_image.depth)
#            plt.show()

#            # Show point cloud of object
#            for j in range(len(pcd5[0])):
#                mb = p.createMultiBody(baseMass=0,
#                           baseCollisionShapeIndex=collisionShapeId,
#                           baseVisualShapeIndex=visualShapeId,
#                           basePosition=pcd5[:3, j]+np.array([0,0.2,0]))



            ##########

            # Set contact points and compute gripper position 
#            contact_point1 = pcd5[:3, 50]
#            contact_point2 = pcd5[:3, 120]
#            contact_point3 = pcd5[:3, 230]
            contact_point1 = np.array([0.28, 0.008, 0.025])  # Test points for banana
            contact_point2 = np.array([0.31, -0.008, 0.025])
            contact_point3 = np.array([0.34, 0.008, 0.025])
#            print(contact_point1, contact_point2, contact_point3)
            CentrePoint, finger_bend, R_CPs_test, R_CPs_test_T = rM.compute_ruth_pose(contact_point1*1000, contact_point2*1000, contact_point3*1000)
            #R_CPs_test: rotation matrix corresponding to contact points
            CentrePoint = CentrePoint/1000 # Compute RUTH position for inverse kinematics
            
            R_x_ax = (contact_point2-contact_point3)/np.linalg.norm(contact_point2-contact_point3)  # x-axis across 'base' points 
            R_y_ax = np.cross(R_x_ax,(contact_point1-contact_point3)/np.linalg.norm(contact_point1-contact_point3))/np.linalg.norm(np.cross(R_x_ax,(contact_point1-contact_point3)/np.linalg.norm(contact_point1-contact_point3)))  # y-axis points normal to plane formed by 3 points
            R_z_ax = np.cross(R_x_ax, R_y_ax)
            R_CPs = np.column_stack((R_x_ax, R_y_ax, R_z_ax))
            approach_distance = 0.3  # distance from which the robot approaches the grasp

            if R_y_ax[2]>0: # If 
#                R_CPs = np.matmul(np.column_stack(([-1,0,0], [0,-1,0], [0,0,1])),R_CPs)
                R_CPs = -R_CPs
#            if R_CPs_test[2, 2]>0: # If z-axis is pointing up
##                R_CPs = np.matmul(np.column_stack(([-1,0,0], [0,-1,0], [0,0,1])),R_CPs)
#                R_CPs_test[:, 0] = -R_CPs_test[:, 0]
#                R_CPs_test[:, 2] = -R_CPs_test[:, 2]


            approach_steps = 500
            approach_step_size = approach_distance/approach_steps
            # CentrePoint_init = CentrePoint - R_CPs_test[:, ]*approach_distance  # approach normal to contact point plane 
            CentrePoint_init = CentrePoint - R_CPs_test[:, 2]*approach_distance  # approach normal to contact point plane 

            R_CPs_to_G = np.column_stack(([-1.,0.,0.], [0.,-1.,0.], [0.,0.,1.]))
            R_CPs2 = np.column_stack((R_CPs_test[:,0], R_CPs_test[:,2], -R_CPs_test[:,1]))

            r = R.from_matrix(R_CPs)
            CPs_Eulers = p.getEulerFromQuaternion(r.as_quat())
            CPs_Quaternion = r.as_quat()

#            sphereA = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point1)
#            sphereB = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point2)
#            sphereC = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point3)
#            p.setCollisionFilterGroupMask(sphereA, -1, 1, 0)
#            p.setCollisionFilterGroupMask(sphereB, -1, 1, 0)
#            p.setCollisionFilterGroupMask(sphereC, -1, 1, 0)
#            con2 = p.createConstraint(parentBodyUniqueId=sphereA,
#                       parentLinkIndex=-1,
#                       childBodyUniqueId=-1,
#                       childLinkIndex=-1,
#                       jointType=p.JOINT_FIXED,
#                       jointAxis=[0, 0, 0],
#                       parentFramePosition=[0, 0, 0],
#                       childFramePosition=contact_point1)
#            con3 = p.createConstraint(parentBodyUniqueId=sphereB,
#                       parentLinkIndex=-1,
#                       childBodyUniqueId=-1,
#                       childLinkIndex=-1,
#                       jointType=p.JOINT_FIXED,
#                       jointAxis=[0, 0, 0],
#                       parentFramePosition=[0, 0, 0],
#                       childFramePosition=contact_point2)
#            con4 = p.createConstraint(parentBodyUniqueId=sphereC,
#                       parentLinkIndex=-1,
#                       childBodyUniqueId=-1,
#                       childLinkIndex=-1,
#                       jointType=p.JOINT_FIXED,
#                       jointAxis=[0, 0, 0],
#                       parentFramePosition=[0, 0, 0],
#                       childFramePosition=contact_point3)



            i = 1

        #############


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
#            ee_pos, ee_ori = ur5_values[:3], p.getQuaternionFromEuler(ur5_values[3:])
            ee_pos, ee_ori = ur5_values[:3], CPs_Quaternion
            R_CPs3 = np.column_stack((-R_CPs2[:,0], -R_CPs2[:,2], -R_CPs2[:,1]))
            R_CPs3 = np.column_stack((-R_CPs_test[:,0], -R_CPs_test[:,1], R_CPs_test[:,2]))
            ee_ori = R.from_matrix(R_CPs3).as_quat()
            # ee_pos = CentrePoint

#=================Start approach object 
            if i<1000:
                if approach_step_count < approach_steps:
                    #ee_pos = CentrePoint_init + R_CPs[:, 1]*approach_step_count*approach_step_size
                    ee_pos = CentrePoint_init + R_CPs_test[:, 2]*approach_step_count*approach_step_size
                    approach_step_count += 1
                else:
                    ee_pos = CentrePoint
            else:
                if i==1000:
                    approach_step_count = 0
                if approach_step_count < approach_steps:
                    #ee_pos = CentrePoint - R_CPs[:, 1]*approach_step_count*approach_step_size
                    ee_pos = CentrePoint - R_CPs_test[:, 2]*approach_step_count*approach_step_size
                    approach_step_count += 1
                else:
                    ee_pos = CentrePoint_init

            lower_joint_limits = [-np.pi]*19
            lower_joint_limits[1] = 0  # So that the shoulder lift joint only goes up   
            upper_joint_limits = [np.pi]*19
#            upper_joint_limits[0] = np.pi/2
            joint_ranges = [2*np.pi]*19
            restposes = [0]*19
#            robot_joints = p.calculateInverseKinematics(robot, linkNameToID['tcp'], ee_pos, ee_ori, lowerLimits=lower_joint_limits, upperLimits=upper_joint_limits, jointRanges=joint_ranges, restPoses=restposes, solver=0, residualThreshold=.01)
            robot_joints = p.calculateInverseKinematics(robot, ur5LinkNameToID['wrist_3_link'], [-0.5, 0, 0.2], (0, 0, 0, 1))
#            print(robot_joints)
#            h/g
            ur5_joints = robot_joints[:6]
#            print(p.getJointState(robot, ur5JointNameToID['shoulder_pan_joint'])[0])
#            print(p.getJointState(robot, ur5JointNameToID['shoulder_lift_joint'])[0])
            ee_pos2 = p.getLinkState(robot, ur5LinkNameToID['wrist_3_link'])
            ee_pos3 = p.getLinkState(robot, linkNameToID['tcp'])

#            print('here')
#            print(ee_pos)
#            print(ee_pos3[0])


        if i<600:
            i+=1
            finger_pos_plus = 0
            bend_incr = 0.5
        else:
            i+=1
            bend_incr = 0
            finger_pos_plus = finger_bend


#        targetPos1 = motor_positions[0]
#        targetPos2 = motor_positions[1]
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, targetPos1, force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, targetPos2, force=100)  
        finger_angles = FA.calc_finger_angles()
        finger_angles = [0, 0, 0]
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_1A'], p.POSITION_CONTROL, -finger_angles[0], force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_2A'], p.POSITION_CONTROL, finger_angles[1], force=100)  
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_3A'], p.POSITION_CONTROL, finger_angles[2], force=100)


        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_1B'], p.POSITION_CONTROL, -0.55+bend_incr-finger_position-finger_pos_plus, force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_1C'], p.POSITION_CONTROL, 1+bend_incr-finger_position-finger_pos_plus, force=100)  
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_2B'], p.POSITION_CONTROL, -0.55+bend_incr-finger_position-finger_pos_plus, force=100)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_2C'], p.POSITION_CONTROL, 1+bend_incr-finger_position-finger_pos_plus, force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_3B'], p.POSITION_CONTROL, 0.65-bend_incr+finger_position+finger_pos_plus, force=100)  
        p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_3C'], p.POSITION_CONTROL, -1-bend_incr+finger_position+finger_pos_plus, force=100)

        p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_pan_joint'], p.POSITION_CONTROL, ur5_joints[0], force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_lift_joint'], p.POSITION_CONTROL, ur5_joints[1], force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['elbow_joint'], p.POSITION_CONTROL, ur5_joints[2], force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['wrist_1_joint'], p.POSITION_CONTROL, ur5_joints[3], force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['wrist_2_joint'], p.POSITION_CONTROL, ur5_joints[4], force=1000)
        p.setJointMotorControl2(robot, ur5JointNameToID['wrist_3_joint'], p.POSITION_CONTROL, ur5_joints[5], force=1000)

#        print('here')
#        print(p.getLinkState(robot, ur5LinkNameToID['wrist_3_link'])[0])
#        print(p.getLinkState(robot, ur5LinkNameToID['wrist_3_link'])[0])
#        print(ur5_joints)
#        print(p.getJointState(robot, ur5JointNameToID['shoulder_pan_joint'])[0])
#        print(p.getJointState(robot, ur5JointNameToID['shoulder_lift_joint'])[0])
#        print(p.getJointState(robot, ur5JointNameToID['elbow_joint'])[0])
#        print(p.getJointState(robot, ur5JointNameToID['wrist_1_joint'])[0])
#        print(p.getJointState(robot, ur5JointNameToID['wrist_2_joint'])[0])
#        print(p.getJointState(robot, ur5JointNameToID['wrist_3_joint'])[0])

        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()


if __name__ == "__main__":
    main()



