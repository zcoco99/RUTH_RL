# Created on 2020.12.20 by Wyn
# A example demo of widowx_arm robot 

import pybullet as p
import numpy as np
import time

p.connect(p.GUI)
plane = p.loadURDF("plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./200)
urdfFlags = p.URDF_USE_INERTIA_FROM_FILE
arm = p.loadURDF("urdf/widowx_arm.urdf",[0,0,0.05],[0,0,0,1], flags = urdfFlags,useFixedBase=True,globalScaling=1.0)

# Arm's info and ID
jointIds=[]
for j in range (p.getNumJoints(arm)):
    p.changeDynamics(arm,j,linearDamping=0, angularDamping=0)
    info = p.getJointInfo(arm,j)
    print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
        jointIds.append(j)
print(jointIds)
#jointIds=(1,2,3,4,5,7,8,9)


while(1):
    targetPos=np.zeros(8)
    lowlimit=np.zeros(8)
    highlimit=np.zeros(8)
    anglerange=np.zeros(8)

    # # Mode 1
    # --- Control all widowx arm joints together
    for j in range(len(jointIds)):  # len(jointIds)=8

        c = jointIds[j] #get 8 joints' id
        # set joint target angle
        lowlimit[j]=( p.getJointInfo(arm,c)[8])
        highlimit[j]=(p.getJointInfo(arm,c)[9])
        anglerange[j]=0.5*(highlimit[j]-lowlimit[j])
        targetPos=0.2*anglerange*np.sin(time.time())
        # set gripper state
        targetPos[6]=0.8*anglerange[6]*np.sin(time.time())
        targetPos[7]=0.8*anglerange[7]*np.sin(time.time())
        if targetPos[6]<0 or targetPos[7]<0:
            targetPos[6]=0.0
            targetPos[7]=0.0
        # excute action command
        # targetPos=[0,1.57,-1.57,0,0,0,0.027,0.027]
        p.setJointMotorControlArray(arm, jointIds, p.POSITION_CONTROL, targetPos)



    # # Mode 2
    # # ---control widowx arm joints individually---
    # p.setJointMotorControl2(arm,1,p.POSITION_CONTROL,0.0,force=300)
    # p.setJointMotorControl2(arm,2,p.POSITION_CONTROL,-1.50,force=300)
    # p.setJointMotorControl2(arm,3,p.POSITION_CONTROL,1.0,force=200)
    # p.setJointMotorControl2(arm,4,p.POSITION_CONTROL,1.0,force=200)
    # p.setJointMotorControl2(arm,5,p.POSITION_CONTROL,0.9,force=150)
    # #grippers
    # p.setJointMotorControl2(arm,7,p.POSITION_CONTROL,0.0,force=0.0)
    # p.setJointMotorControl2(arm,8,p.POSITION_CONTROL,0.027,force=150)
    # p.setJointMotorControl2(arm,9,p.POSITION_CONTROL,0.027,force=150)

    print(targetPos, highlimit, lowlimit)
    p.setRealTimeSimulation(1)
    time.sleep(.001)

