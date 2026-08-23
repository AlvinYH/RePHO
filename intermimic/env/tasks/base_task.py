# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import os
import operator
from copy import deepcopy
import random

from isaacgym import gymapi, gymtorch
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

import numpy as np
import torch
import time


# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, enable_camera_sensors=False):
        self.gym = gymapi.acquire_gym()
        self.cfg = cfg

        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        #automatically get from cuda_visible_devices
        
        print("enable_camera_sensors",cfg["enable_camera_sensors"])
        enable_camera_sensors = cfg.get("enable_camera_sensors", False)
        self.enable_camera_sensors = enable_camera_sensors
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
            # self.device = "cuda"

        self.headless = cfg["headless"]
        self.cam_params_path = os.path.join(self.root_file_path, self.sub_file_name, 'cam_intermimic.npz')

        self.graphics_device_id = self.device_id
        print('self.graphics_device_id',self.graphics_device_id)

        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg["env"]["numEnvs"]
        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]

        # This is a benchmark clock parameter, not an optional convenience
        # setting.  A missing value must never silently turn into 60 Hz.
        self.control_freq_inv = cfg["env"]["controlFrequencyInv"]

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.start_times = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.data_id = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        self.original_props = {}
        self.dr_randomizations = {}
        self.first_randomization = True
        self.actor_params_generator = None
        self.extern_actor_params = {}
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.last_step = -1
        self.last_rand_step = -1
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 90.0
            camera_props.width = 400
            camera_props.height = 224
            # viewer = gym.create_viewer(sim, camera_props)
            self.viewer = self.gym.create_viewer(
                self.sim, camera_props)
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)
        self.render_rgb = cfg["env"].get("render_rgb", False)
        self.render_depth = cfg["env"].get("render_depth", False)
        self.render_seg = cfg["env"].get("render_seg", False)
        if self.enable_camera_sensors:
            self.num_envs_with_cam = self.num_envs
            self.create_camera_sensors()



    def project_points_to_camera(self, batch_points_3d):
        """Project 3D points to 2D pixel coordinates using camera matrices"""
        bsz,num_points, _ = batch_points_3d.shape
        points_3d = batch_points_3d.view(-1, 3)
        # 1. Convert to homogeneous coordinates (N,4)
        points_h = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=1)
        # 2. Apply view matrix to transform to camera space
        points_cam = (points_h @ self.camera_view_matrix)
        # 3. Apply projection matrix to get normalized device coordinates
        points_ndc = (points_cam @ self.camera_proj_matrix)
        # 4. Perspective divide to get 2D coordinates
        points_2d = points_ndc[:, :2] / points_ndc[:, 3:4]
        # 5. Convert from NDC [-1,1] to pixel coordinates [0,W]x[0,H]
        points_pixel = torch.zeros_like(points_2d)
        points_pixel[:, 0] = (points_2d[:, 0] + 1.0) * 400 * 0.5  # width=1600
        points_pixel[:, 1] = (points_2d[:, 1] + 1.0) * 224 * 0.5   # height=900
        points_pixel = points_pixel.view(bsz, num_points, 2)
        points_2d = points_2d.view(bsz, num_points, 2)
        return points_2d, points_pixel
    

    def create_camera_sensors(self):
        """Create camera sensors for headless rendering"""
        self.gym.prepare_sim(self.sim)
        camera_props = gymapi.CameraProperties()
        if self.mode == 'train':
            camera_props.width = 960
            camera_props.height = 540
        else:
            camera_props.width = 1920
            camera_props.height = 1080

        if self.cam_params_path:
            cam_params = np.load(self.cam_params_path, allow_pickle=True)['arr_0'].item()
        # camera_props.horizontal_fov = 93.4
        # Calculate horizontal FOV from camera intrinsic matrix K
        # FOV = 2 * arctan(width / (2 * focal_length))
        print(cam_params['K'])  # K is a 3x3 matrix
        focal_length = cam_params['K'][0,0]  # Get focal length from K matrix
        camera_props.horizontal_fov = 2 * np.arctan(cam_params['K'][0,2] / focal_length) * 180 / np.pi * 1.0
        # Enable GPU tensor output for the camera
        camera_props.enable_tensors = True
        self.camera_images_buffer = []
        self.camera_segs_buffer = []
        self.camera_depths_buffer = []

        self.camera_handles = []
        for i in range(self.num_envs_with_cam):
            # Create camera sensor for each environment
            camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
            self.camera_handles.append(camera_handle)
            
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                print('up axis is z')
                start_point = cam_params['start_point']
                end_point = cam_params['end_point']
                # start_point[0] = -start_point[0]
                # end_point[0] = -end_point[0]
                # start_point[1] = -start_point[1]
                # end_point[1] = -end_point[1]

                # cam_pos = gymapi.Vec3(-0.096318,2.47350874,1.12232364)
                # cam_target = gymapi.Vec3(-0.05498641,1.47517669,1.16263278)
                cam_pos = gymapi.Vec3(start_point[0],start_point[1],start_point[2])
                cam_target = gymapi.Vec3(end_point[0],end_point[1],end_point[2])
                # cam_pos = gymapi.Vec3(-2, 2, 1)
                # cam_target = gymapi.Vec3(0, 0.0, 1)
            else:
                print('up axis is y')
                # cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                # cam_target = gymapi.Vec3(10.0, 0.0, 15.0)
                cam_pos = gymapi.Vec3(2.5, 1.0, 2.5)
                cam_target = gymapi.Vec3(0.0, 1.0, 0.0)
            self.gym.set_camera_location(camera_handle, self.envs[i], cam_pos, cam_target)
        view_matrix = self.gym.get_camera_view_matrix(self.sim, self.envs[0], self.camera_handles[0])  
        proj_matrix = self.gym.get_camera_proj_matrix(self.sim, self.envs[0], self.camera_handles[0])  
        self.camera_proj_matrix = torch.tensor(proj_matrix, device=self.device)
        self.camera_view_matrix = torch.tensor(view_matrix, device=self.device)
        for idx in range(self.num_envs_with_cam):
            if self.render_rgb:
                self.camera_images_buffer.append(gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[idx], self.camera_handles[idx], gymapi.IMAGE_COLOR)))
            if self.render_seg or (self.reward_2d and (self.reward_weights.get('o_2d_maskany',0)>0)):
                self.camera_segs_buffer.append(gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[idx], self.camera_handles[idx], gymapi.IMAGE_SEGMENTATION)))
            if self.render_depth:
                self.camera_depths_buffer.append(gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[idx], self.camera_handles[idx], gymapi.IMAGE_DEPTH)))


    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            gravity = self.cfg["env"]["gravity"]
            sim_params.gravity.x = gravity[0]
            sim_params.gravity.y = gravity[1]
            sim_params.gravity.z = gravity[2]
            return 2
        return 1

    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):

        compute_device = self.device_id
        graphics_device = self.graphics_device_id
        print(compute_device, graphics_device, 'compute_device, graphics_device')
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def step(self, actions):
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    def get_states(self):
        return self.states_buf

    def render_camera_sensors(self):
        """Render and save images using camera sensors in headless mode"""
        if not hasattr(self, 'camera_handles'):
            return
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        if self.render_rgb:
            self.camera_image_tensors = torch.stack(self.camera_images_buffer, dim=0)
        if self.render_seg or (self.reward_2d and self.reward_weights.get('o_2d_maskany',0)>0):
            self.camera_seg_tensors = torch.stack(self.camera_segs_buffer, dim=0)
        if self.render_depth:
            self.camera_depth_tensors = torch.stack(self.camera_depths_buffer, dim=0)

        # self.gym.refresh_camera_image_tensors(self.sim)

        self.gym.end_access_image_tensors(self.sim)
        # print(self.camera_image_tensors.shape, self.camera_seg_tensors.shape)

        return 

    # This render method handles visualization and graphics updates for the simulation
    def render(self, sync_frame_time=False):
        # Only execute if there is a viewer
        if self.viewer:
            # Exit the program if the viewer window is closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # Handle keyboard input events:
            # - QUIT: Exit the program
            # - toggle_viewer_sync: Switch between synchronized and unsynchronized rendering
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # For GPU simulation, fetch results before rendering
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # Update graphics:
            # If sync enabled: Update and draw graphics in sync with simulation
            # If sync disabled: Just poll for viewer events without updating graphics
            if self.enable_viewer_sync:
                # step_graphics updates the physics simulation's visual representation 
                # by advancing the graphics engine one step forward, ensuring the 
                # visual state matches the current physics state
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)


    def get_actor_params_info(self, dr_params, env):
        """Returns a flat array of actor params, their names and ranges."""
        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name+'_'+str(prop_idx)+'_'+attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0*float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name+'_'+str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    # Apply randomizations only on resets, due to current PhysX limitations
    def apply_randomizations(self, dr_params):
        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        for actor, actor_properties in dr_params["actor_params"].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue
                    if prop_name == 'scale':
                        attr_randomization_params = prop_attrs
                        sample = generate_random_samples(attr_randomization_params, 1,
                                                         self.last_step, None)
                        og_scale = 1
                        if attr_randomization_params['operation'] == 'scaling':
                            new_scale = og_scale * sample
                        elif attr_randomization_params['operation'] == 'additive':
                            new_scale = og_scale + sample
                        self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], p, attr)
                                apply_random_samples(
                                    p, og_p, attr, attr_randomization_params,
                                    self.last_step, smpl)
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            smpl = None
                            if self.actor_params_generator is not None:
                                smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                    extern_sample, extern_offsets[env_id], prop, attr)
                            apply_random_samples(
                                prop, self.original_props[prop_name], attr,
                                attr_randomization_params, self.last_step, smpl)

                    setter = param_setters_map[prop_name]
                    default_args = param_setter_defaults_map[prop_name]
                    setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:

                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def _physics_step(self):
        for i in range(self.control_freq_inv):
            self.render()
            time_in = time.time()
            self.gym.simulate(self.sim)
            # print('phys_time', time.time() - time_in)
        return

    def post_physics_step(self):
        raise NotImplementedError


def get_attr_val_from_sample(sample, offset, prop, attr):
    """Retrieves param value for the given prop and attr from the sample."""
    if sample is None:
        return None, 0
    if isinstance(prop, np.ndarray):
        smpl = sample[offset:offset+prop[attr].shape[0]]
        return smpl, offset+prop[attr].shape[0]
    else:
        return sample[offset], offset+1
