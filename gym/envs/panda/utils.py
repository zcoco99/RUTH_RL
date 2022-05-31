import numpy as np


def ctrl_set_action(bullet_client, panda, pandaNumDofs, action, force):
    for i in range(pandaNumDofs):
        bullet_client.setJointMotorControl2(panda, i, bullet_client.POSITION_CONTROL, action[i], force=force)
    for i in range(2):
        bullet_client.setJointMotorControl2(panda, i+pandaNumDofs+2, bullet_client.POSITION_CONTROL, action[i+pandaNumDofs],force=force)
    
    

def robot_get_obs(bullet_client, panda, pandaNumDofs):
    obs_pos = []
    obs_vel = []
    for i in range(pandaNumDofs):
        getJoint = bullet_client.getJointState(panda,i)
        obs_pos.append(getJoint[0])
        obs_vel.append(getJoint[1])
    grip_joint_left = bullet_client.getJointState(panda,pandaNumDofs+2)
    grip_joint_right = bullet_client.getJointState(panda,pandaNumDofs+3)
    obs_pos.append(grip_joint_left[0]); obs_pos.append(grip_joint_right[0])
    obs_vel.append(grip_joint_left[1]); obs_vel.append(grip_joint_right[1])
    
    return np.array(obs_pos), np.array(obs_vel), np.array([grip_joint_left[0]])
    

def robot_get_achievedgoal(bullet_client, panda, pandaNumDofs):
    
    # grip_link_left = bullet_client.getLinkState(panda,pandaNumDofs+2)
    # grip_link_right = bullet_client.getLinkState(panda,pandaNumDofs+3)
    # achieved_goal = (np.array(grip_link_left[0]) + np.array(grip_link_right[0]))/2
    # return achieved_goal
    
    grip_link = bullet_client.getLinkState(panda,pandaNumDofs+4)
    return np.array(grip_link[0])
    