import numpy as np

from gym.envs.robotics import rotations, robot_env, utils
import cv2

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PileEnv(robot_env.RobotEnv):
    """Superclass for all Pile environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        object_num, target_in_the_air, obs_index, initial_qpos, reward_type,
    ):
        """Initializes a new Pile environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            object_num (int): the number of objects in the environment
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.object_num = [object_num]
        self.target_in_the_air = target_in_the_air
        self.object_range = initial_qpos['object']
        self.obs_index = obs_index

        self.reward_type = reward_type
        
        self.is_target = [False]
        self.size = 1

        super(PileEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

#    def compute_reward(self, achieved_goal, goal, info):
#        # Compute distance between goal and the achieved goal.
#        if self.reward_type == 'sparse':
#            return 0
#        else:
#            return 0


    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
        
    
    def resetgoal(self, omega, goal, tempgoalcolor=None):
        return None

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        objects_pos, objects_rot, objects_velp, objects_velr, objects_rel_pos = [], [], [], [], []
        for i in range(self.object_num[0]):
            object_name = 'object'+str(i)
            object_pos = self.sim.data.get_site_xpos(object_name)
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(object_name))
            # velocities
            object_velp = self.sim.data.get_site_xvelp(object_name) * dt
            object_velr = self.sim.data.get_site_xvelr(object_name) * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
            
            objects_pos.append(object_pos)
            objects_rot.append(object_rot)
            objects_velp.append(object_velp)
            objects_velr.append(object_velr)
            objects_rel_pos.append(object_rel_pos)
            
        objects_pos = np.array(objects_pos)
        objects_rot = np.array(objects_rot)
        objects_velp = np.array(objects_velp)
        objects_velr = np.array(objects_velr)
        objects_rel_pos = np.array(objects_rel_pos)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        
#        img = self.render_frame(camera_id=-1, imagesize=(50,50)).copy()
        
        img = cv2.resize(self.render_frame(camera_id=-1, imagesize=(500,500)).copy()[100:-200, 150:-200], (50,50))
        
#        if self.object_num[0] == 0:
#            return [np.concatenate([
#                    np.concatenate([grip_pos, gripper_state]).copy(), #5
#                    ]), img]
        return [np.concatenate([
                robot_qpos.copy(), 
                grip_pos.copy(), #5
                ]), img]
        
#        achieved_goal = np.squeeze(object_pos.copy())
#        return [np.concatenate([
##                achieved_goal.copy()[:2], #3
##                np.concatenate([objects_pos, objects_rot], -1).reshape(-1).copy(),#[self.obs_index], #18
##                np.concatenate([robot_qpos, robot_qvel]).copy(), 
#                grip_pos, 
#                robot_qpos, #15
##                np.concatenate([grip_pos, gripper_state]).copy(),
##                np.concatenate([grip_pos, gripper_state]).copy(), #5
##                achieved_goal.copy() #3
#                ]), None]

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        pass

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        for i in range(self.object_num[0]):
            name = 'object'+str(i)+':joint'
            value = self.object_range['object:joint:center'].copy()
#            obj_range = self.object_range['object:joint:range'][:3]
#            obj_size = self.object_range['object:joint:size'][:3]
##            self.seed(i)
#            value[:3] = value[:3] + self.np_random.uniform(-np.array(obj_range)+np.array(obj_size), np.array(obj_range)-np.array(obj_size), size=3)
            self.sim.data.set_joint_qpos(name, value)
        
        
        if self.is_target[0]:
#            name = 'target0'
            value = self.object_range['object:joint:center'].copy()
##            obj_range = self.object_range['object:joint:range']
##            obj_size = self.object_range['object:joint:size']
##            value[:3] = value[:3] + self.np_random.uniform(-np.array(obj_range)+np.array(obj_size), np.array(obj_range)-np.array(obj_size), size=3)
#            self.sim.data.set_joint_qpos(name, value)
            
            site_id = self.sim.model.site_name2id('target0')
            self.sim.model.site_pos[site_id] = value[:3]
            
            

        self.sim.forward()
        return True

    def _sample_goal(self):
        return None

    def _is_success(self, achieved_goal, desired_goal):
        return None

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos['robot'].items():
            self.sim.data.set_joint_qpos(name, value)
        for i in range(self.object_num[0]):
            name = 'object'+str(i)+':joint'
            value = initial_qpos['object']['object:joint:center'].copy()
            self.sim.data.set_joint_qpos(name, value)
        
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        # self.sim.data.set_mocap_pos('line0:mocap', gripper_target)
        # self.sim.data.set_mocap_quat('line0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.object_num[0]>0:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(PileEnv, self).render(mode, width, height)
