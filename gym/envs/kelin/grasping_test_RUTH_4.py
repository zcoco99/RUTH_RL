import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pybullet as p
import numpy as np
import time
import pybullet_data
from scipy.spatial.transform import Rotation as R
from ruth_grasping_kinematics import ruthModel
from pybullet_object_models import ycb_objects
import open3d as o3d
import matplotlib.pyplot as plt


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


def visualise_contact_points(contact_point1, contact_point2, contact_point3):  # Create spheres to visualise positions of contact points
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
    visualShapeId = -1
    sphereA = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point1)
    sphereB = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point2)
    sphereC = p.createMultiBody(0.01, colSphereId, visualShapeId, contact_point3)
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


def set_ur5_position(robot, ur5JointNameToID, ur5_joints):  # set positions of ur5 joints
    p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_pan_joint'], p.POSITION_CONTROL, ur5_joints[0],
                            force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['shoulder_lift_joint'], p.POSITION_CONTROL, ur5_joints[1],
                            force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['elbow_joint'], p.POSITION_CONTROL, ur5_joints[2], force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['wrist_1_joint'], p.POSITION_CONTROL, ur5_joints[3], force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['wrist_2_joint'], p.POSITION_CONTROL, ur5_joints[4], force=1000)
    p.setJointMotorControl2(robot, ur5JointNameToID['wrist_3_joint'], p.POSITION_CONTROL, ur5_joints[5], force=1000)


def set_ruth_position(robot, jointNameToID, ruth_all_vals):
    p.setJointMotorControl2(robot, jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, ruth_all_vals[0], force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, ruth_all_vals[1], force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_1A'], p.POSITION_CONTROL, ruth_all_vals[2], force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_2A'], p.POSITION_CONTROL, ruth_all_vals[3], force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_3A'], p.POSITION_CONTROL, ruth_all_vals[4], force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_1B'], p.POSITION_CONTROL, ruth_all_vals[5], force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_1C'], p.POSITION_CONTROL, ruth_all_vals[6], force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_2B'], p.POSITION_CONTROL, ruth_all_vals[7], force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_2C'], p.POSITION_CONTROL, ruth_all_vals[8], force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_3B'], p.POSITION_CONTROL, ruth_all_vals[9], force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Phal_3C'], p.POSITION_CONTROL, ruth_all_vals[10], force=1000)

rM = ruthModel()


def main():

    physics_client = p.connect(p.GUI)  # for p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -9.8)

    urdf_directory = "../urdf/ur5_plus_RUTH.urdf"
    # urdf_root = pybullet_data.getDataPath()

    FixedBase = False #if fixed no plane is imported
    if (FixedBase == False):
        floor = p.loadURDF("plane.urdf")

    robotPos = [0, 0, 0.2]
    robotScale = 1
    robot = p.loadURDF(urdf_directory,
                       robotPos,
                       p.getQuaternionFromEuler([0, 0, 0]),
                       useFixedBase=1,
                       globalScaling=robotScale)

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

    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01 * robotScale)
    visualShapeId = -1

    # Disable Collisions between links
    for link in linkNameToID:
        p.setCollisionFilterGroupMask(robot, linkNameToID[link], 1, 0)
    for link in ur5LinkNameToID:
        p.setCollisionFilterGroupMask(robot, ur5LinkNameToID[link], 1, 0)
    p.setCollisionFilterGroupMask(robot, -1, 1, 0) 
    p.setCollisionFilterGroupMask(floor, -1, 1, 0)

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

    p.setJointMotorControl2(robot, jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, 0.0, force=1000)
    p.setJointMotorControl2(robot, jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, 0.0, force=1000)

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

    top = np.tan(fov_rad/2)*near
    bottom = -top
    right = top*aspect_ratio
    left = -top*aspect_ratio
    proj_mat = np.array([[2*near/(right-left), 0, (right+left)/(right-left), 0], [0, 2*near/(top-bottom), (top+bottom)/(top-bottom), 0], [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)], [0, 0, -1, 0]])
    
    for i in range(1000):
        p.stepSimulation()
        time.sleep(1./240.)

    p.setRealTimeSimulation(0)
    targetPos1 = 0.0
    targetPos2 = 0.0
    FA = FingerAngles(robot, linkNameToID)  # Finger angles class

    # Loop just to initialise robot
    for i in range(1000):
        targetPos1 = -i*np.pi/100000
        targetPos2 = i*np.pi/100000
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_1'], p.POSITION_CONTROL, targetPos1, force=1000)
        p.setJointMotorControl2(robot, jointNameToID['Joint_Link_3'], p.POSITION_CONTROL, targetPos2, force=1000)
        set_ur5_position(robot, ur5JointNameToID, [0, np.pi/2, 0, 0, 0, 0])
        p.stepSimulation()

    i = 0
    approach_step_count = 0
    ur5_values = [0]*6
    finger_position = 0

    while True:

        # If first loop, get camera image and make rgbd image
        if i == 0:
            flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=image_width,
                height=image_height,
                viewMatrix=viewMatrix,
                projectionMatrix=projectionMatrix, flags=flags)
            depthImgb = far * near / (far - (far-near)*depthImg)
            rgbImg2 = o3d.geometry.Image((np.array(rgbImg)[:, :, :3]).astype(np.uint8))
            depthImg2 = o3d.geometry.Image(depthImgb)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgbImg2, depthImg2)

    #        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #    rgbd_image,
    #    o3d.camera.PinholeCameraIntrinsic(
    #        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(image_width, image_height, f, f, image_width/2, image_height/2))

            pcd_points = np.asarray(pcd.points)
            ii = 0
            pcd2 = np.empty((0, 3))
            rgbd_acc = np.zeros((image_width, image_height))
            print(np.asarray(rgbImg).shape)
            rgb_test = np.zeros((image_width, image_height, 3))
            for j in range(image_width):
                for k in range(image_height):
                    rgbd_acc[j][k] = ii/image_width**2
                    ii += 1
                    if np.asarray(segImg[j][k]) == obj:
                        rgb_test[j][k][:] = 255
                        pcd2 = np.vstack((pcd2, pcd_points[ii,:]))  # Collect point cloud oordinates which correspond to object
            pcd3 = o3d.geometry.PointCloud()
            pcd3.points = o3d.utility.Vector3dVector(pcd2)
            pcd4 = np.vstack((np.transpose(-pcd2*1000), np.ones(len(pcd2))))  # Homogenous coordinates and scale up by depth scale (1000)
            view_matrix2 = np.transpose(np.asarray(viewMatrix).reshape((4,4)))
            pcd5 = np.matmul(np.linalg.inv(view_matrix2), pcd4)  # convert from point cloud coordinates to world coordinates
            pcd5[1, :] = -pcd5[1, :]

            # print(pcd2.shape)
            # o3d.visualization.draw_geometries([pcd3])  # Plot point cloud in Open3D
            
            sample_rate = int(np.floor(pcd2.shape[0]/2048))
            pcd6 = o3d.geometry.PointCloud.uniform_down_sample(pcd3, sample_rate)
            # print(np.asarray(pcd6.points).shape, sample_rate)
            # o3d.visualization.draw_geometries([pcd6])  # Plot point cloud in Open3D

            sample_rate = 2048/np.asarray(pcd6.points).shape[0]
            pcd6 = pcd6.random_down_sample(sample_rate)
            # print(np.asarray(pcd6.points).shape, sample_rate)
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


            # Set contact points and compute gripper position 
#            contact_point1 = pcd5[:3, 50]
#            contact_point2 = pcd5[:3, 120]
#            contact_point3 = pcd5[:3, 230]
            contact_point1 = np.array([0.28, 0.008, 0.025])  # Test points for banana
            contact_point2 = np.array([0.31, -0.008, 0.025])
            contact_point3 = np.array([0.34, 0.008, 0.025])
            CentrePoint, finger_bend, R_CPs_test, R_CPs_test_T = rM.compute_ruth_pose(contact_point1*1000, contact_point2*1000, contact_point3*1000)
            CentrePoint = CentrePoint/1000 # Compute RUTH position for inverse kinematics
            
            R_x_ax = (contact_point2-contact_point3)/np.linalg.norm(contact_point2-contact_point3)  # x-axis across 'base' points 
            R_y_ax = np.cross(R_x_ax,(contact_point1-contact_point3)/np.linalg.norm(contact_point1-contact_point3))/np.linalg.norm(np.cross(R_x_ax,(contact_point1-contact_point3)/np.linalg.norm(contact_point1-contact_point3)))  # y-axis points normal to plane formed by 3 points
            R_z_ax = np.cross(R_x_ax, R_y_ax)
            R_CPs = np.column_stack((R_x_ax, R_y_ax, R_z_ax))
            approach_distance = 0.3  # distance from which the robot approaches the grasp

            if R_y_ax[2] > 0:  # If
#                R_CPs = np.matmul(np.column_stack(([-1,0,0], [0,-1,0], [0,0,1])),R_CPs)
                R_CPs = -R_CPs
#            if R_CPs_test[2, 2]>0: # If z-axis is pointing up
##                R_CPs = np.matmul(np.column_stack(([-1,0,0], [0,-1,0], [0,0,1])),R_CPs)
#                R_CPs_test[:, 0] = -R_CPs_test[:, 0]
#                R_CPs_test[:, 2] = -R_CPs_test[:, 2]

            # Set gripper approach point
            approach_steps = 500
            approach_step_size = approach_distance/approach_steps
            CentrePoint_init = CentrePoint - R_CPs_test[:, 2]*approach_distance  # approach normal to contact point plane

            r = R.from_matrix(R_CPs)
            CPs_Eulers = p.getEulerFromQuaternion(r.as_quat())

            # visualise_contact_points(contact_point1, contact_point2, contact_point3)

            i = 1

        #############

        ur5_values[3] = CPs_Eulers[0]  # + ur5_values[3]
        ur5_values[4] = CPs_Eulers[2]  # + ur5_values[4]
        ur5_values[5] = CPs_Eulers[1]  # + ur5_values[5]
        R_CPs3 = np.column_stack((-R_CPs_test[:, 0], -R_CPs_test[:, 1], R_CPs_test[:, 2]))
        ee_ori = R.from_matrix(R_CPs3).as_quat()
        if i < 1000:  # First 1000 steps, moves towards grasping position
            if approach_step_count < approach_steps:
                #ee_pos = CentrePoint_init + R_CPs[:, 1]*approach_step_count*approach_step_size
                ee_pos = CentrePoint_init + R_CPs_test[:, 2]*approach_step_count*approach_step_size
                approach_step_count += 1
            else:
                ee_pos = CentrePoint
        else:  # Once grasped, move beck to approach point
            if i == 1000:
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
        joint_ranges = [2*np.pi]*19
        restposes = [0]*19
        robot_joints = p.calculateInverseKinematics(robot, linkNameToID['tcp'], ee_pos, ee_ori, lowerLimits=lower_joint_limits, upperLimits=upper_joint_limits, jointRanges=joint_ranges, restPoses=restposes, solver=0, residualThreshold=.01)
#            robot_joints = p.calculateInverseKinematics(robot, ur5LinkNameToID['wrist_3_link'], [-0.5, 0, 0.2], (0, 0, 0, 1))
#            print(robot_joints)
#            h/g
        ur5_joints = robot_joints[:6]

        if i < 600:  # Before grasping
            i += 1
            finger_pos_plus = 0
            bend_incr = 0.5
        else:  # Instruct gripper to grasp
            i += 1
            bend_incr = 0
            finger_pos_plus = finger_bend

        five_bar_vals = [targetPos1, targetPos2]
        finger_angles = FA.calc_finger_angles()
        finger_angles = [0, 0, 0]
        finger_ori_vals = [-finger_angles[0], finger_angles[1], finger_angles[2]]
        finger_bend_vals = [-0.55+bend_incr-finger_position-finger_pos_plus, 1+bend_incr-finger_position-finger_pos_plus, -0.55+bend_incr-finger_position-finger_pos_plus, 1+bend_incr-finger_position-finger_pos_plus, 0.65-bend_incr+finger_position+finger_pos_plus, -1-bend_incr+finger_position+finger_pos_plus]
        ruth_all_vals = five_bar_vals + finger_ori_vals + finger_bend_vals

        set_ruth_position(robot, jointNameToID, ruth_all_vals)
        set_ur5_position(robot, ur5JointNameToID, ur5_joints[:6])


        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()


if __name__ == "__main__":
    main()



