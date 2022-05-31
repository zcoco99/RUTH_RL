import math
import numpy as np

import pybullet as p
import pybullet_data as pd

from gym.envs.panda import robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, version, offset, pandaNumDofs, n_substeps, force_substeps, 
        camera, initial_qpos, 
        has_object, block_gripper, 
        target_range, distance_threshold,
        reward_type, 
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments urdf file
            offset (array): 
            n_substeps (int): number of substeps the simulation runs on every call to step
            camera (array): 
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            has_object (boolean): whether or not the environment has an object
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            target_range (float): range of a uniform distribution for sampling a target 
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense 
            distance_threshold (float): the threshold after which a goal is considered achieved
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        if version == 'GUI':
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
        elif version == 'DIRECT':
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pd.getDataPath())
        #planeId = p.loadURDF("plane.urdf")
        
        p.setGravity(0,-9.8,0)
        
        self.bullet_client = p
        self.pandaNumDofs = pandaNumDofs
        self.pandaActionNum = 3 # x y z
        
        orn=[-0.707107, 0.0, 0.0, 0.707107]
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.panda = self.bullet_client.loadURDF(model_path, np.array([0,0,0])+offset, orn, useFixedBase=True, flags=flags)
        
        euler_gripper=[math.pi/2., 0., 0.]
        self.orn_gripper = self.bullet_client.getQuaternionFromEuler(euler_gripper)
        self.lluljr={'ll':[-7]*pandaNumDofs, 'ul':[7]*pandaNumDofs, 'jr':[7]*pandaNumDofs}
        
        if camera is not None:
            self.bullet_client.resetDebugVisualizerCamera(camera[0],camera[1],camera[2],camera[3])
        visualShapeId = self.bullet_client.createVisualShape(shapeType=self.bullet_client.GEOM_SPHERE,
                                    radius = 0.05,
                                    rgbaColor=[1, 0, 0, 1])
        self.ball = self.bullet_client.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=visualShapeId,
                          basePosition=[0,0,0],
                            useMaximalCoordinates=True)
        
        self.current_state = np.zeros(self.pandaActionNum+1)
        self.goal = np.array([0,0,0])
        
        self.n_substeps = n_substeps
        self.force_substeps = force_substeps
        
        self.initial_qpos = initial_qpos
        self.has_object = has_object
        self.block_gripper = block_gripper
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super(FetchEnv, self).__init__(
            n_actions=self.pandaActionNum+1, ) # x y z gripper


    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self._is_done(achieved_goal):
            return -10.0 
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d


    def _step_callback(self):
        # if self.block_gripper:
        #     for i in range(2):
        #         self.bullet_client.setJointMotorControl2(self.panda, i+self.pandaNumDofs+2, 
        #                     self.bullet_client.POSITION_CONTROL, 0.02,force=self.force_substeps)
        pass
            
        

    def _set_action(self, action):
        assert action.shape == (self.pandaActionNum+1,) 
        action = action * 0.5
        action = action + self.current_state 
        # action = np.array(self.target_range[1])
        actionJoint = self.bullet_client.calculateInverseKinematics(self.panda, 11, action[:self.pandaActionNum], self.orn_gripper, 
                                                                    self.lluljr['ll'], self.lluljr['ul'], self.lluljr['jr'], 
                                                                    self.initial_qpos, maxNumIterations=self.n_substeps)
        actionJoint = np.array(actionJoint)
        if self.block_gripper:
            actionJoint[-2:] = np.zeros(2)+0.02
        else:
            gripper_ctrl = action[self.pandaNumDofs]
            actionJoint[-2:] = np.array([gripper_ctrl, gripper_ctrl])
        # Apply actionJoint to simulation.
        utils.ctrl_set_action(self.bullet_client, self.panda, self.pandaNumDofs, actionJoint, self.force_substeps)
        
        # for i in range(self.pandaNumDofs):
        #     self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, action[i], force=self.force_substeps)
        

    def _get_obs(self):
        obs_pos, obs_vel, gripper = utils.robot_get_obs(self.bullet_client, self.panda, self.pandaNumDofs)
        achieved_goal = utils.robot_get_achievedgoal(self.bullet_client, self.panda, self.pandaNumDofs)
        obs = np.concatenate([obs_pos, obs_vel], 0)
        self.current_state = np.concatenate([achieved_goal, gripper], 0).copy()
        return {
            # 'observation': obs.copy(),
            # 'observation': achieved_goal.copy(),
            'observation': np.concatenate([obs.copy(), achieved_goal.copy()], 0), 
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
        # return [None, self.bullet_client.getCameraImage(width=50, height=50)]
    
    

    def _viewer_setup(self):
        pass

    def _render_callback(self):
        pass

    def _reset_sim(self):
        
        for i in range(self.bullet_client.getNumJoints(self.panda)):
          self.bullet_client.changeDynamics(self.panda, i, linearDamping=0, angularDamping=0)
          info = self.bullet_client.getJointInfo(self.panda, i)
          jointName, jointType = info[1], info[2]
          if (jointType == self.bullet_client.JOINT_PRISMATIC):       
            self.bullet_client.resetJointState(self.panda, i, self.initial_qpos[i])
          if (jointType == self.bullet_client.JOINT_REVOLUTE):
            self.bullet_client.resetJointState(self.panda, i, self.initial_qpos[i]) 
        
        # Randomize start position of object.
        if self.has_object:
            self.goal = self.np_random.uniform(self.target_range[0], self.target_range[1])
            # self.goal = np.array([0.66,0.44,0.22])
            self.bullet_client.resetBasePositionAndOrientation(self.ball, self.goal, [0,0,0,1])
            
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.goal 
        else:
            goal = np.array([0,0,0])
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
    
    def _is_done(self, achieved_goal, margine=0.1):
        x, y, z = achieved_goal
        donex =  x<self.target_range[0][0]-margine or x>self.target_range[1][0]+margine 
        doney =  y<self.target_range[0][1]-margine or y>self.target_range[1][1]+margine 
        donez =  z<self.target_range[0][2]-margine or z>self.target_range[1][2]+margine 
        done = donex or doney or donez
        return done
        

    def _env_setup(self, initial_qpos):
        pass

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
