from enum import Enum
import numpy as np
import torch
import os

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils
import torch.nn.functional as F
from env.tasks.humanoid import *
import trimesh
import cv2

import imageio
import time
import json
import xml.etree.ElementTree as ET



class InterMimic(Humanoid_SMPLX):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3
        Traverse = 4
        History = 5
        Traverse_Random = 6


    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.dual_update = True
        self.thin_obj = cfg['env'].get('thin_obj', False)
        self.mode = cfg['env']['mode']
        state_init = cfg["env"]["stateInit"]
        self._state_init = InterMimic.StateInit[state_init]
        self._init_range_left = cfg["env"].get("init_range_left", None)
        self._init_range_right = cfg["env"].get("init_range_right", None)
        self.reverse_time = cfg["env"].get("reverse_time", False)
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self.save_states = cfg["env"].get("save_states", False)
        self.disable_gravity = cfg["env"].get("disable_gravity", False)
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self.motion_file = cfg['env']['motion_file']
        self.root_file_path = cfg['env']['motion_file']
        self.sub_file_name = cfg['env'].get("sub_file_name", 'intermimic')
        self.hoi_refs_path = cfg['env'].get("hoi_refs_path", None)
        self.hoi_data_path = cfg['env'].get("hoi_data_path", None)
        self.file_name = self.motion_file.split('/')[-1]
        self.play_dataset = cfg['env']['playdataset']
        if self.play_dataset:
            cfg["env"]["disableSelfCollision"] = True
            self._extra_agg_bodies = 52
            self._extra_agg_shapes = 52
        else:
            self._extra_agg_bodies = 0
            self._extra_agg_shapes = 0
        self.reward_weights = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages']
        self.init_vel = cfg['env']['initVel']
        self.ball_size = cfg['env']['ballSize']
        self.more_rigid = cfg['env']['moreRigid']
        self.rollout_length = cfg['env']['rolloutLength']
        self.psi = cfg['env'].get('physicalBufferSize', 1)
        self.object_name = [self.file_name.split('_')[2]]
        self.robot_type = os.path.join(self.motion_file, self.sub_file_name, 'intermimic_humanoid.xml')
        self.motion_file = [os.path.join(self.motion_file, self.sub_file_name, 'intermimic.pt')]
        object_name_set = sorted(list(set(self.object_name)))
        self.device = "cuda" + ":" + str(cfg.get("device_id", 0))
        self.object_id = to_torch([object_name_set.index(name) for name in self.object_name], dtype=torch.long, device=self.device)
        self.obj2motion = torch.stack([self.object_id == k for k in range(len(object_name_set))], dim=0)
        self.object_name = object_name_set
        self.object_density = cfg['env']['objectDensity']
        self.ref_hoi_obs_size = 7 + 51 * 6 + 52 * 13 + 13 + 52 * 3 + 52 + 1
        self.num_motions = len(self.motion_file)
        self.dataset_index = to_torch([0], dtype=torch.long, device=self.device)
        self.reward_2d = cfg['env']['reward_2d']

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.object_id = self.object_id.to(self.device)
        self.dataset_index = self.dataset_index.to(self.device)
        self.hoi_data = self._load_motion(self.motion_file, topk=self.psi)
        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.kinematic_reset = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.contact_reset = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.dataset_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._curr_reward = torch.zeros([self.num_envs, self.max_episode_length.max()], device=self.device, dtype=torch.float)
        self._sum_reward = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._curr_state = torch.zeros([self.num_envs, self.max_episode_length.max(), 332], device=self.device, dtype=torch.float)
        powers = torch.arange(self.max_episode_length.max(), device=self.device)
        self.powers = 0.99 ** powers
        self._curr_state_complement = torch.zeros([self.num_envs, self.max_episode_length.max(), 676], device=self.device, dtype=torch.float)
        self._curr_contact = torch.zeros([self.num_envs, self.max_episode_length.max(), 99], device=self.device, dtype=torch.float)
        self._curr_contact[:,:,:3] = 1

        self._init_range_left = 0 if self._init_range_left is None else self._init_range_left
        self._init_range_right = self.max_episode_length.max().item() - 24 if self._init_range_right is None else self._init_range_right

        self.cam_img_dir = cfg['env']['cam_img_dir']
        self.load_traj_cnt = 0
        self.just_update_tar = False
        self._finish_epoch = 59000

        if self._state_init == InterMimic.StateInit.History:
            stage1_path = os.path.dirname(self.cam_img_dir) + '/stage1_test/combine_play_length_hist.npz'
            self.play_length_hist_origin = torch.from_numpy(np.load(stage1_path, allow_pickle=True)['play_length_hist'])
            if self.reverse_time:
                self.play_length_hist_origin = self.play_length_hist_origin.flip(0)
            self.init_prob_dist = self.play_length_hist_origin[self._init_range_left:self._init_range_right]/self.play_length_hist_origin[self._init_range_left:self._init_range_right].sum()
            self.init_prob_dist = self.init_prob_dist.to(self.device)

        if self._state_init == InterMimic.StateInit.Traverse_Random:
            self._init_valid  = torch.from_numpy(np.load(os.path.join(self.root_file_path, self.sub_file_name, 'penetration.npy'))).to(self.device)

            if self.reverse_time:
                self._init_valid = self._init_valid.flip(0)
            self.ref_reward[:,:,self._init_valid] = 0
            self._init_valid = torch.logical_not(self._init_valid)
            self._init_valid = torch.where(self._init_valid[self._init_range_left:self._init_range_right])[0]+self._init_range_left

            random_idx = torch.cat([torch.randperm(self._init_valid.size(0)) for _ in range(20)], dim=0).to(self.device)
            self.Traverse_Random_Queue = self._init_valid[random_idx]


        self._build_target_tensors()
        self.cam_img_dir = cfg['env']['cam_img_dir']
        self.contact_obj_whole = self.extract_data_component('contact_obj', obs=self.hoi_data[0, 0:self.max_episode_length[0]])

        if self.enable_camera_sensors and self.reward_2d:
            if not self.play_dataset and self.reward_weights.get('p_2d', 0) > 0:
                pure_2d_key_list = []
                self.pure_2d_key_path = os.path.join(self.root_file_path, 'vitpose')
                # Get list of all .pt files and sort them by frame number
                pt_files = sorted([f for f in os.listdir(self.pure_2d_key_path) if f.endswith('.pt')], 
                                key=lambda x: int(x.split('frame')[-1].split('.')[0]))
                
                # Load files in order
                for file in pt_files:
                    pure_2d_key_list.append(torch.load(os.path.join(self.pure_2d_key_path, file)))
                self.pure_2d_key = torch.stack(pure_2d_key_list, dim=0).to(self.device)

            if not self.play_dataset and self.reward_weights.get('o_2d_maskany', 0) > 0:
                obj_mask_list = []
                self.obj_mask_dir = os.path.join(self.root_file_path, 'masks')
                # Get list of all png files and sort them by frame number
                mask_files = sorted([f for f in os.listdir(self.obj_mask_dir) if f.endswith('.png')],
                                key=lambda x: int(x.split('frame')[-1].split('.')[0]))
                # Load files in order
                for file in mask_files:

                    obj_mask_list.append(torch.from_numpy(imageio.imread(os.path.join(self.obj_mask_dir, file))))
                self.ref_obj_mask = (torch.stack(obj_mask_list, dim=0)/255).unsqueeze(1)
                self.ref_obj_mask = F.interpolate(self.ref_obj_mask, size=(60, 80), mode='nearest').squeeze(1)
                self.ref_obj_mask = self.ref_obj_mask.to(torch.float16).to(self.device)
        return

    def post_physics_step(self):
        super().post_physics_step()
        if self.enable_camera_sensors and self.save_images:
            self.render_camera_sensors()
            if self.save_images:
                if not hasattr(self, 't_before'):
                    self.t_before = 0
                if self.render_rgb:
                    self.save_cam_imgs()
                if self.render_seg or (self.reward_2d and self.reward_weight.get('o_2d_maskany',0)>0):
                    self.save_cam_segs()
                self.t_before+=1
        return

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return

    def _load_new_motion(self, data_path, startk=0, topk=1, initk=0):
        loaded_dict = {}
        if not self.reverse_time:
            hoi_data = torch.load(data_path,map_location=self.device)[startk:]
        else:
            hoi_data = torch.load(data_path,map_location=self.device).flip(0)[startk:]

        loaded_dict['hoi_data'] = hoi_data.detach().to(self.device)

        max_episode_length=[loaded_dict['hoi_data'].shape[0]]
        self.fps_data = 30.

        loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
        loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:].clone() - loaded_dict['root_pos'][:-1,:].clone())*self.fps_data
        loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])).to(self.device),loaded_dict['root_pos_vel']),dim=0)

        loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7].clone()
        root_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['root_rot'])
        loaded_dict['root_rot_vel'] = (root_rot_exp_map[1:,:].clone() - root_rot_exp_map[:-1,:].clone())*self.fps_data
        loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to(self.device),loaded_dict['root_rot_vel']),dim=0)

        loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+153].clone()

        loaded_dict['dof_vel'] = []

        loaded_dict['dof_vel'] = (loaded_dict['dof_pos'][1:,:].clone() - loaded_dict['dof_pos'][:-1,:].clone())*self.fps_data
        loaded_dict['dof_vel'] = torch.cat((torch.zeros((1, loaded_dict['dof_vel'].shape[-1])).to(self.device),loaded_dict['dof_vel']),dim=0)

        loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 162: 162+52*3].clone()
        loaded_dict['body_pos_vel'] = (loaded_dict['body_pos'][1:,:].clone() - loaded_dict['body_pos'][:-1,:].clone())*self.fps_data
        loaded_dict['body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_pos_vel'].shape[-1])).to(self.device),loaded_dict['body_pos_vel']),dim=0)

        loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 318:321].clone()

        loaded_dict['obj_pos_vel'] = (loaded_dict['obj_pos'][1:,:].clone() - loaded_dict['obj_pos'][:-1,:].clone())*self.fps_data
        if self.init_vel:
            loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
        else:
            loaded_dict['obj_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_pos_vel'].shape[-1])).to(self.device),loaded_dict['obj_pos_vel']),dim=0)


        loaded_dict['obj_rot'] = loaded_dict['hoi_data'][:, 321:325].clone()
        obj_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['obj_rot'])
        loaded_dict['obj_rot_vel'] = (obj_rot_exp_map[1:,:].clone() - obj_rot_exp_map[:-1,:].clone())*self.fps_data
        loaded_dict['obj_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_rot_vel'].shape[-1])).to(self.device),loaded_dict['obj_rot_vel']),dim=0)


        obj_rot_extend = loaded_dict['obj_rot'].unsqueeze(1).repeat(1, self.object_points[self.object_id[0]].shape[0], 1).view(-1, 4)
        object_points_extend = self.object_points[self.object_id[0]].unsqueeze(0).repeat(loaded_dict['obj_rot'].shape[0], 1, 1).view(-1, 3)

        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict['obj_rot'].shape[0], self.object_points[self.object_id[0]].shape[0], 3) + loaded_dict['obj_pos'].unsqueeze(1)

        ref_ig = compute_sdf(loaded_dict['body_pos'].view(max_episode_length[-1],52,3), obj_points).view(-1, 3)
        heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['root_rot'])
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, loaded_dict['body_pos'].shape[1] // 3, 1).view(-1, 4)
        ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)    
        loaded_dict['ig'] = ref_ig
        loaded_dict['contact_obj'] = torch.round(loaded_dict['hoi_data'][:, 330:331].clone())
        loaded_dict['contact_human'] = torch.round(loaded_dict['hoi_data'][:, 331:331+52].clone())

        loaded_dict['body_rot'] = loaded_dict['hoi_data'][:, 331+52:331+52+52*4].clone()
    
        human_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['body_rot'].view(-1, 4)).view(-1, 52*3)
        loaded_dict['body_rot_vel'] = (human_rot_exp_map[1:,:].clone() - human_rot_exp_map[:-1,:].clone())*self.fps_data
        loaded_dict['body_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_rot_vel'].shape[-1])).to(self.device),loaded_dict['body_rot_vel']),dim=0)
        loaded_dict['hoi_data'] = torch.cat((
                                                loaded_dict['root_pos'].clone(), #b,3
                                                loaded_dict['root_rot'].clone(), #b,4 
                                                loaded_dict['dof_pos'].clone(), #b,153
                                                loaded_dict['dof_vel'].clone(), #b,153
                                                loaded_dict['body_pos'].clone(), #b,156
                                                loaded_dict['body_rot'].clone(), #b,208
                                                loaded_dict['body_pos_vel'].clone(), #b,156
                                                loaded_dict['body_rot_vel'].clone(), #b,156
                                                loaded_dict['obj_pos'].clone(),  #b,3
                                                loaded_dict['obj_rot'].clone(), #b,4
                                                loaded_dict['obj_pos_vel'].clone(),   #b,3
                                                loaded_dict['obj_rot_vel'].clone(),   #b,3
                                                loaded_dict['ig'].clone(), #b,156
                                                loaded_dict['contact_human'].clone(), #b,52
                                                loaded_dict['contact_obj'].clone(), #b,1
                                                ),dim=-1)
        return loaded_dict['hoi_data']
    

    def _load_motion(self, motion_file, startk=0, topk=1, initk=0):
        self.curr_epoch = 53001
        hoi_datas = []
        hoi_refs = []
        if type(motion_file) != type([]):
            motion_file = [motion_file]
        max_episode_length = []
        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            print(data_path)

            if not self.reverse_time:
                hoi_data = torch.load(data_path)[startk:]
            else:
                hoi_data = torch.load(data_path).flip(0)[startk:]
                print('reverse_time')

            loaded_dict['hoi_data'] = hoi_data.detach().to(self.device)

            max_episode_length.append(loaded_dict['hoi_data'].shape[0])
            self.fps_data = 30.

            loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
            loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:].clone() - loaded_dict['root_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])).to(self.device),loaded_dict['root_pos_vel']),dim=0)

            loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7].clone()
            root_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['root_rot'])
            loaded_dict['root_rot_vel'] = (root_rot_exp_map[1:,:].clone() - root_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to(self.device),loaded_dict['root_rot_vel']),dim=0)

            loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+153].clone()

            loaded_dict['dof_vel'] = []

            loaded_dict['dof_vel'] = (loaded_dict['dof_pos'][1:,:].clone() - loaded_dict['dof_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['dof_vel'] = torch.cat((torch.zeros((1, loaded_dict['dof_vel'].shape[-1])).to(self.device),loaded_dict['dof_vel']),dim=0)

            loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 162: 162+52*3].clone()
            loaded_dict['body_pos_vel'] = (loaded_dict['body_pos'][1:,:].clone() - loaded_dict['body_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_pos_vel'].shape[-1])).to(self.device),loaded_dict['body_pos_vel']),dim=0)

            loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 318:321].clone()

            loaded_dict['obj_pos_vel'] = (loaded_dict['obj_pos'][1:,:].clone() - loaded_dict['obj_pos'][:-1,:].clone())*self.fps_data
            if self.init_vel:
                loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
            else:
                loaded_dict['obj_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_pos_vel'].shape[-1])).to(self.device),loaded_dict['obj_pos_vel']),dim=0)


            loaded_dict['obj_rot'] = loaded_dict['hoi_data'][:, 321:325].clone()
            obj_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['obj_rot'])
            loaded_dict['obj_rot_vel'] = (obj_rot_exp_map[1:,:].clone() - obj_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['obj_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_rot_vel'].shape[-1])).to(self.device),loaded_dict['obj_rot_vel']),dim=0)


            obj_rot_extend = loaded_dict['obj_rot'].unsqueeze(1).repeat(1, self.object_points[self.object_id[idx]].shape[0], 1).view(-1, 4)
            object_points_extend = self.object_points[self.object_id[idx]].unsqueeze(0).repeat(loaded_dict['obj_rot'].shape[0], 1, 1).view(-1, 3)

            obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict['obj_rot'].shape[0], self.object_points[self.object_id[idx]].shape[0], 3) + loaded_dict['obj_pos'].unsqueeze(1)

            ref_ig = compute_sdf(loaded_dict['body_pos'].view(max_episode_length[-1],52,3), obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['root_rot'])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, loaded_dict['body_pos'].shape[1] // 3, 1).view(-1, 4)
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)    
            loaded_dict['ig'] = ref_ig
            loaded_dict['contact_obj'] = torch.round(loaded_dict['hoi_data'][:, 330:331].clone())
            
            loaded_dict['contact_human'] = torch.round(loaded_dict['hoi_data'][:, 331:331+52].clone())

            self.left_hand_ids = list(range(17, 33))
            self.right_hand_ids = list(range(36, 52))
            self.contact_label_left_hand = torch.any(loaded_dict['contact_human'][:, self.left_hand_ids]>0.5, dim=-1)
            self.contact_label_right_hand = torch.any(loaded_dict['contact_human'][:, self.right_hand_ids]>0.5, dim=-1)

            loaded_dict['body_rot'] = loaded_dict['hoi_data'][:, 331+52:331+52+52*4].clone()
        
            human_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['body_rot'].view(-1, 4)).view(-1, 52*3)
            loaded_dict['body_rot_vel'] = (human_rot_exp_map[1:,:].clone() - human_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['body_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_rot_vel'].shape[-1])).to(self.device),loaded_dict['body_rot_vel']),dim=0)

            loaded_dict['hoi_data'] = torch.cat((
                                                    loaded_dict['root_pos'].clone(), #b,3
                                                    loaded_dict['root_rot'].clone(), #b,4 
                                                    loaded_dict['dof_pos'].clone(), #b,153
                                                    loaded_dict['dof_vel'].clone(), #b,153
                                                    loaded_dict['body_pos'].clone(), #b,156
                                                    loaded_dict['body_rot'].clone(), #b,208
                                                    loaded_dict['body_pos_vel'].clone(), #b,156
                                                    loaded_dict['body_rot_vel'].clone(), #b,156
                                                    loaded_dict['obj_pos'].clone(),  #b,3
                                                    loaded_dict['obj_rot'].clone(), #b,4
                                                    loaded_dict['obj_pos_vel'].clone(),   #b,3
                                                    loaded_dict['obj_rot_vel'].clone(),   #b,3
                                                    loaded_dict['ig'].clone(), #b,156
                                                    loaded_dict['contact_human'].clone(), #b,52
                                                    loaded_dict['contact_obj'].clone(), #b,1
                                                    ),dim=-1)

            assert(self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1])
            loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'][0:1] for _ in range(initk)]+[loaded_dict['hoi_data']], dim=0)
            hoi_datas.append(loaded_dict['hoi_data'])

            hoi_ref = torch.cat((
                                loaded_dict['root_pos'].clone(), 
                                loaded_dict['root_rot'].clone(), 
                                loaded_dict['root_pos_vel'].clone(),
                                loaded_dict['root_rot_vel'].clone(), 
                                loaded_dict['dof_pos'].clone(), 
                                loaded_dict['dof_vel'].clone(), 
                                loaded_dict['obj_pos'].clone(),
                                loaded_dict['obj_rot'].clone(),
                                loaded_dict['obj_pos_vel'].clone(),
                                loaded_dict['obj_rot_vel'].clone(),
                                ),dim=-1)
            hoi_refs.append(hoi_ref)
        max_length = max(max_episode_length) + initk
        self.num_motions = len(hoi_refs)

        self.max_episode_length = to_torch(max_episode_length, dtype=torch.long) + initk
        self.max_episode_length = self.max_episode_length.to(self.device)
        hoi_data = []
        self.hoi_refs = []
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            hoi_data.append(padded_data)
            self.hoi_refs.append(F.pad(hoi_refs[i], pad_size, "constant", 0))
        hoi_data = torch.stack(hoi_data, dim=0)
        self.hoi_refs = torch.stack(self.hoi_refs, dim=0).unsqueeze(1).repeat(1, topk, 1, 1)
        self.ref_reward = torch.zeros((self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2])).to(self.hoi_refs.device)
        self.ref_reward_sum = self.ref_reward.clone()

        self.ref_reward[:, 0, :] = 1.0
        self.contact_refs = torch.zeros((self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2], 99)).to(self.hoi_refs.device)
        self.contact_refs[:, :, :, :3] = 0 #valid bit
        if self.hoi_refs_path is not None:
            self.hoi_refs = np.load(self.hoi_refs_path, allow_pickle=True)['hoi_refs']
            self.hoi_refs = torch.from_numpy(self.hoi_refs).to(self.device)
            self.hoi_refs = self.hoi_refs[:,1,:,:].repeat(1, topk, 1, 1)
            if self.reverse_time:
                self.hoi_refs = self.hoi_refs.flip(-2)
                velocity_indices = list(range(7, 13)) + list(range(166, 319)) + list(range(326, 332))
                self.hoi_refs[:,:, :, velocity_indices] *= -1
            self.contact_refs = self.hoi_refs[:,:, :, -99:].clone()
            self.hoi_refs = self.hoi_refs[:,:, :, :-99]
        

        self.ref_reward_for_opposite = self.ref_reward.clone()
        self.hoi_refs_for_opposite = self.hoi_refs.clone()
        self.contact_refs_for_opposite = self.contact_refs.clone()
        self.ref_traj = torch.zeros((self.hoi_refs.shape[0], 5, self.hoi_refs.shape[2], 1002)).to(self.hoi_refs.device) #1002 = 1211-52-1-156
        self.ref_traj_reward = torch.zeros((self.hoi_refs.shape[0], 5)).to(self.hoi_refs.device)

        self.to_end_cnt = 0
        self.middle_to_end_cnt = 0
        self.left_to_end_cnt = 0

        self.ref_index = torch.zeros((self.num_envs, )).long().to(self.hoi_refs.device)
        if not hasattr(self, 'data_component_order'):
            self.create_component_stat(loaded_dict)

        if self.hoi_data_path is not None:
            hoi_data[0] = torch.load(self.hoi_data_path, map_location=self.device)#.flip(0)
        return hoi_data.contiguous()

    def create_component_stat(self, loaded_dict):
        self.data_component_order = [
            'root_pos', 'root_rot', 'dof_pos', 'dof_vel', 'body_pos', 'body_rot', 'body_pos_vel', 'body_rot_vel',
            'obj_pos', 'obj_rot', 'obj_pos_vel', 'obj_rot_vel', 'ig', 'contact_human', 'contact_obj'
        ]

        # Precompute the sizes for each component.
        data_component_sizes = [
            loaded_dict[name].shape[1]
            for name in self.data_component_order
        ]

        # Precompute cumulative indices. The first index is zero.
        # For each i, calculate the sum of component_sizes[:i] to determine the starting index for that component.
        self.data_component_index = [sum(data_component_sizes[:i]) for i in range(len(data_component_sizes) + 1)]

        self.ref_component_order = [
            'root_pos', 'root_rot', 'root_pos_vel', 'root_rot_vel', 'dof_pos', 'dof_vel', 'obj_pos', 'obj_rot', 
            'obj_pos_vel', 'obj_rot_vel'
        ]

        # Precompute the sizes for each component.
        ref_component_sizes = [
            loaded_dict[name].shape[1]
            for name in self.ref_component_order
        ]

        # Precompute cumulative indices. The first index is zero.
        # For each i, calculate the sum of component_sizes[:i] to determine the starting index for that component.
        self.ref_component_index = [sum(ref_component_sizes[:i]) for i in range(len(ref_component_sizes) + 1)]


    def extract_ref_component(self, var_name, data_id, ref_index, t):
        index = self.ref_component_order.index(var_name)
        
        # The number of columns to extract for this component.
        start = self.ref_component_index[index]
        end = self.ref_component_index[index+1]
        
        return self.hoi_refs[data_id, ref_index, t, start:end]


    def extract_data_component(self, var_name, ref=False, data_id=None, t=None, obs=None):
        index = self.data_component_order.index(var_name)
        
        # The number of columns to extract for this component.
        start = self.data_component_index[index]
        end = self.data_component_index[index+1]
        
        if ref and data_id is not None and t is not None:
            return self.hoi_data[data_id, t, start:end]
        
        if obs is not None:
            return obs[..., start:end]

    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._body_proxy_handles = []
        self._load_target_asset()
        if self.play_dataset:
            self._load_play_dataset_body_proxy_assets()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if self.play_dataset:
            self.gym.set_actor_scale(env_ptr, self.humanoid_handles[env_id], 1e-4)

        self._build_target(env_id, env_ptr)
        if self.play_dataset:
            self._build_play_dataset_body_proxies(env_id, env_ptr)
        return   

    def _load_play_dataset_body_proxy_assets(self):
        self._body_proxy_asset = []
        self._body_proxy_names = []
        local_pos = []
        local_rot = []

        source_xml = os.path.join(self.root_file_path, self.sub_file_name, 'intermimic_humanoid.xml')
        tree = ET.parse(source_xml)
        root = tree.getroot()
        body_nodes = []

        def visit_body(node):
            body_nodes.append(node)
            for child in node.findall('body'):
                visit_body(child)

        worldbody = root.find('worldbody')
        for body in worldbody.findall('body'):
            visit_body(body)

        if len(body_nodes) != self._extra_agg_bodies:
            raise RuntimeError(
                'Expected {} humanoid body proxies, found {} in {}'.format(
                    self._extra_agg_bodies, len(body_nodes), source_xml)
            )

        proxy_asset_root = os.path.join(
            self.root_file_path, self.sub_file_name, 'play_dataset_body_proxy_assets')
        os.makedirs(proxy_asset_root, exist_ok=True)

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = False

        for body_idx, body in enumerate(body_nodes):
            geom = body.find('geom')
            if geom is None:
                raise RuntimeError('Body {} has no geom in {}'.format(body.attrib.get('name', body_idx), source_xml))

            body_name = body.attrib.get('name', 'body_{}'.format(body_idx))
            geom_type = geom.attrib.get('type', 'sphere')
            self._body_proxy_names.append(body_name)

            if geom_type == 'sphere':
                offset = self._parse_vec3_attr(geom.attrib.get('pos', '0 0 0'))
                rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            elif geom_type == 'box':
                offset = self._parse_vec3_attr(geom.attrib.get('pos', '0 0 0'))
                rot = self._parse_mjcf_quat_attr(geom.attrib.get('quat', '1 0 0 0'))
            elif geom_type == 'capsule':
                fromto = np.array([float(v) for v in geom.attrib['fromto'].split()], dtype=np.float32)
                start = fromto[:3]
                end = fromto[3:]
                axis = end - start
                length = float(np.linalg.norm(axis))
                if length <= 1e-8:
                    raise RuntimeError('Capsule body {} has degenerate fromto in {}'.format(body_name, source_xml))
                offset = 0.5 * (start + end)
                rot = self._quat_from_axis_to_vec(np.array([0.0, 0.0, 1.0], dtype=np.float32), axis / length)
            else:
                raise RuntimeError('Unsupported play_dataset proxy geom type {} on body {}'.format(geom_type, body_name))

            proxy_asset_file = self._write_play_dataset_body_proxy_asset(
                proxy_asset_root, body_idx, body_name, geom)
            asset = self.gym.load_asset(
                self.sim, proxy_asset_root, os.path.basename(proxy_asset_file), asset_options)
            body_count = self.gym.get_asset_rigid_body_count(asset)
            shape_count = self.gym.get_asset_rigid_shape_count(asset)
            if body_count != 1 or shape_count != 1:
                raise RuntimeError(
                    'Expected one body and one shape in proxy asset {}, got {} bodies and {} shapes'.format(
                        proxy_asset_file, body_count, shape_count)
                )
            self._body_proxy_asset.append(asset)
            local_pos.append(offset)
            local_rot.append(rot)

        self._body_proxy_local_pos = to_torch(np.stack(local_pos, axis=0), device=self.device, dtype=torch.float)
        self._body_proxy_local_rot = to_torch(np.stack(local_rot, axis=0), device=self.device, dtype=torch.float)
        return

    def _write_play_dataset_body_proxy_asset(self, asset_root, body_idx, body_name, geom):
        safe_name = self._safe_asset_name(body_name)
        mesh_name = 'proxy_{:02d}_{}.obj'.format(body_idx, safe_name)
        urdf_name = 'proxy_{:02d}_{}.urdf'.format(body_idx, safe_name)
        mesh_path = os.path.join(asset_root, mesh_name)
        urdf_path = os.path.join(asset_root, urdf_name)

        mesh = self._create_play_dataset_body_proxy_mesh(geom)
        mesh.export(mesh_path)

        robot = ET.Element('robot', {'name': 'proxy_{:02d}_{}'.format(body_idx, safe_name)})
        link = ET.SubElement(robot, 'link', {'name': 'proxy_body'})
        visual = ET.SubElement(link, 'visual')
        ET.SubElement(visual, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
        visual_geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(visual_geometry, 'mesh', {'filename': mesh_name})
        material = ET.SubElement(visual, 'material', {'name': 'proxy_body_mat'})
        ET.SubElement(material, 'color', {'rgba': '0.75 0.54 0.30 1'})

        collision = ET.SubElement(link, 'collision')
        ET.SubElement(collision, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
        collision_geometry = ET.SubElement(collision, 'geometry')
        ET.SubElement(collision_geometry, 'sphere', {'radius': '0.001'})

        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
        ET.SubElement(inertial, 'mass', {'value': '0.001'})
        ET.SubElement(inertial, 'inertia', {
            'ixx': '1e-6', 'ixy': '0', 'ixz': '0',
            'iyy': '1e-6', 'iyz': '0', 'izz': '1e-6',
        })

        tree = ET.ElementTree(robot)
        tree.write(urdf_path, encoding='utf-8', xml_declaration=True)
        return urdf_path

    def _create_play_dataset_body_proxy_mesh(self, geom):
        geom_type = geom.attrib.get('type', 'sphere')
        if geom_type == 'sphere':
            radius = float(geom.attrib['size'].split()[0])
            offset = self._parse_vec3_attr(geom.attrib.get('pos', '0 0 0')).astype(np.float64)
            transform = self._make_transform_np(np.eye(3, dtype=np.float64), offset)
            return trimesh.creation.uv_sphere(radius=radius, count=[16, 32], transform=transform)

        if geom_type == 'box':
            size = np.array([float(v) for v in geom.attrib['size'].split()], dtype=np.float64)
            offset = self._parse_vec3_attr(geom.attrib.get('pos', '0 0 0')).astype(np.float64)
            rot = self._parse_mjcf_quat_attr(geom.attrib.get('quat', '1 0 0 0')).astype(np.float64)
            transform = self._make_transform_np(self._quat_xyzw_to_matrix_np(rot), offset)
            return trimesh.creation.box(extents=2.0 * size, transform=transform)

        if geom_type == 'capsule':
            fromto = np.array([float(v) for v in geom.attrib['fromto'].split()], dtype=np.float64)
            start = fromto[:3]
            end = fromto[3:]
            axis = end - start
            length = float(np.linalg.norm(axis))
            if length <= 1e-8:
                raise RuntimeError('Proxy capsule has degenerate fromto {}'.format(geom.attrib['fromto']))
            radius = float(geom.attrib['size'].split()[0])
            offset = 0.5 * (start + end)
            rot = self._rotation_matrix_from_vec_to_vec_np(
                np.array([0.0, 0.0, 1.0], dtype=np.float64), axis / length)
            transform = self._make_transform_np(rot, offset)
            return trimesh.creation.capsule(height=length, radius=radius, count=[12, 24], transform=transform)

        raise RuntimeError('Unsupported play_dataset proxy mesh geom type {}'.format(geom_type))

    def _safe_asset_name(self, name):
        return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)

    def _build_play_dataset_body_proxies(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        default_pose.p = gymapi.Vec3(10000.0, 10000.0, 10000.0)
        default_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        col_group = env_id + 2 * self.num_envs
        col_filter = 1
        segmentation_id = 0

        env_handles = []
        for body_idx, asset in enumerate(self._body_proxy_asset):
            body_name = self._body_proxy_names[body_idx]
            handle = self.gym.create_actor(
                env_ptr, asset, default_pose,
                'play_dataset_body_{:02d}_{}'.format(body_idx, body_name),
                col_group, col_filter, segmentation_id)
            self.gym.set_rigid_body_color(
                env_ptr, handle, 0, gymapi.MESH_VISUAL,
                gymapi.Vec3(0.75, 0.54, 0.3))
            env_handles.append(handle)

        self._body_proxy_handles.append(env_handles)
        return

    def _parse_vec3_attr(self, text):
        return np.array([float(v) for v in text.split()], dtype=np.float32)

    def _parse_mjcf_quat_attr(self, text):
        quat_wxyz = [float(v) for v in text.split()]
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)

    def _quat_from_axis_to_vec(self, source, target):
        source = source.astype(np.float32)
        source = source / np.linalg.norm(source)
        target = target.astype(np.float32)
        target = target / np.linalg.norm(target)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

        if dot > 1.0 - 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if dot < -1.0 + 1e-6:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(source, fallback))) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            axis = np.cross(source, fallback)
            axis = axis / np.linalg.norm(axis)
            return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)

        axis = np.cross(source, target)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(dot)
        sin_half = np.sin(0.5 * angle)
        return np.array([
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            np.cos(0.5 * angle),
        ], dtype=np.float32)

    def _make_transform_np(self, rot, pos):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        return transform

    def _quat_xyzw_to_matrix_np(self, quat):
        quat = np.asarray(quat, dtype=np.float64)
        quat = quat / np.linalg.norm(quat)
        x, y, z, w = quat
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float64)

    def _rotation_matrix_from_vec_to_vec_np(self, source, target):
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        source = source / np.linalg.norm(source)
        target = target / np.linalg.norm(target)
        cross = np.cross(source, target)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

        if dot > 1.0 - 1e-8:
            return np.eye(3, dtype=np.float64)
        if dot < -1.0 + 1e-8:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(source, fallback))) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = np.cross(source, fallback)
            axis = axis / np.linalg.norm(axis)
            return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

        skew = np.array([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + skew + skew @ skew * (1.0 / (1.0 + dot))

    def _load_target_asset(self): # smplx
        
        self._target_asset = []
        points_num = []
        self.object_points = []
        points_list = []
        faces_list = []
        for i, object_name in enumerate(self.object_name):

            asset_file = object_name + ".urdf"
            asset_root = os.path.join(self.root_file_path, object_name)
            obj_file = asset_root+ '/' + object_name + '.obj'
            max_convex_hulls = 64
            density = self.object_density
        
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01

            asset_options.density = density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000
            if self.disable_gravity:
                asset_options.disable_gravity = True


            self._target_asset.append(self.gym.load_asset(self.sim, asset_root, asset_file, asset_options))

            mesh_obj = trimesh.load(obj_file, force='mesh')
            obj_verts = mesh_obj.vertices
            center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)

            object_points = to_torch(object_points - center)
            


            while object_points.shape[0] < 1024:
                object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
            self.object_points.append(to_torch(object_points))

        self.object_points = torch.stack(self.object_points, dim=0).to(self.device)
        
        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id + self.num_envs if self.play_dataset else env_id
        col_filter = 1 if self.play_dataset else 0
        segmentation_id = 1

        default_pose = gymapi.Transform()
        
        target_handle = self.gym.create_actor(env_ptr, self._target_asset[env_id % len(self.object_name)], default_pose, self.object_name[env_id % len(self.object_name)], col_group, col_filter, segmentation_id)

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        for p_idx in range(len(props)):
            props[p_idx].restitution = 0.1 #0.6
            props[p_idx].friction = 0.8 #0.8
            props[p_idx].rolling_friction = 0.01
            props[p_idx].torsion_friction = 0.8
            if self.thin_obj:
                props[p_idx].thickness = 0.002
            if not self.thin_obj:
                # props[p_idx].rest_offset = 0.015
                props[p_idx].rest_offset = 0.015
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props)

        self._target_handles.append(target_handle)
        self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)

        return

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        root_states_view = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])
        self._target_states = root_states_view[..., 1, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1

        if self.play_dataset:
            proxy_actor_offsets = torch.arange(self.num_bodies, device=self.device, dtype=torch.int32) + 2
            proxy_actor_bases = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32).unsqueeze(1)
            self._body_proxy_actor_ids = proxy_actor_bases + proxy_actor_offsets.unsqueeze(0)
            self._body_proxy_states = root_states_view[:, 2:2 + self.num_bodies, :]
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]
        return
    
    def _reset_target(self, env_ids):
        self._target_states[env_ids, :3] = self.extract_ref_component('obj_pos', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 3:7] = self.extract_ref_component('obj_rot', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 7:10] = self.extract_ref_component('obj_pos_vel', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 10:13] = self.extract_ref_component('obj_rot_vel', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        return  

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)


        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
        return

    def save_curr_epoch(self, epoch):
        self.curr_epoch = epoch
        
        
    def save_ref_reward(self, epoch):
        self.curr_epoch = epoch
        if epoch >56000:
            print('run too long, quit')
            quit()
        if self.curr_epoch < 53200 or self.curr_epoch % 100 == 0:
            to_draw = True
        else:
            to_draw = False
        os.makedirs(self.cam_img_dir+'/ref_reward', exist_ok=True)
        os.makedirs(self.cam_img_dir+'/ref_hoi', exist_ok=True)
        os.makedirs(self.cam_img_dir+'/ref_contact', exist_ok=True)
        file_path = os.path.join(self.cam_img_dir, 'ref_reward',f'ref_reward_{epoch}.npz')
        file_path_for_opposite = os.path.join(self.cam_img_dir, 'ref_reward',f'opposite_ref_reward_{epoch}.npz')
        file_path_for_hoi = os.path.join(self.cam_img_dir, 'ref_hoi',f'ref_hoi_{epoch}.npz')
        file_path_for_contact = os.path.join(self.cam_img_dir, 'ref_contact',f'ref_contact_{epoch}.npz')
        file_path_for_hoi_opposite = os.path.join(self.cam_img_dir, 'ref_hoi',f'opposite_ref_hoi_{epoch}.npz')
        indices_for_velocity = list(range(7, 13)) + list(range(166, 319)) + list(range(326, 332))
        if self.reverse_time:
            np.savez(file_path, ref_reward=self.ref_reward.flip(2).cpu().numpy())

            np.savez(file_path_for_opposite, ref_reward=self.ref_reward_for_opposite.flip(2).cpu().numpy())


            hoi_refs_for_opposite_for_save = torch.cat([self.hoi_refs_for_opposite.clone(),self.contact_refs_for_opposite.clone()], dim=-1)
            
            hoi_refs_for_opposite_for_save[:,:, :, indices_for_velocity] *= -1
            
            hoi_refs_for_save = torch.cat([self.hoi_refs.clone(),self.contact_refs.clone()], dim=-1)
            hoi_refs_for_save[:, :, :, indices_for_velocity] *= -1
            np.savez(file_path_for_hoi_opposite, hoi_refs=hoi_refs_for_opposite_for_save.flip(2).cpu().numpy())
            np.savez(file_path_for_hoi, hoi_refs=hoi_refs_for_save.flip(2).cpu().numpy())
            np.savez(file_path_for_contact, contact_refs=self.contact_refs.flip(2).cpu().numpy()[:,:,140:170])
        else:
            np.savez(file_path, ref_reward=self.ref_reward.cpu().numpy())

            np.savez(file_path_for_opposite, ref_reward=self.ref_reward_for_opposite.cpu().numpy())


            np.savez(file_path_for_hoi_opposite, hoi_refs=self.hoi_refs_for_opposite.cpu().numpy())
            np.savez(file_path_for_hoi_opposite, hoi_refs=torch.cat([self.hoi_refs_for_opposite.clone(),self.contact_refs_for_opposite.clone()], dim=-1).cpu().numpy())
            np.savez(file_path_for_hoi, hoi_refs=self.hoi_refs.cpu().numpy())
            np.savez(file_path_for_hoi, hoi_refs=torch.cat([self.hoi_refs.clone(),self.contact_refs.clone()], dim=-1).cpu().numpy())
            np.savez(file_path_for_contact, contact_refs=self.contact_refs.cpu().numpy()[:,:,140:170])
        if epoch % 100 == 0:
            os.makedirs(self.cam_img_dir+'/hoi_data', exist_ok=True)

            torch.save(self.hoi_data[0].cpu(), os.path.join(self.cam_img_dir, 'hoi_data',f'intermimic_{epoch}.pt'))
        if epoch % 100 == 0:
            self.save_run_val()
            if (epoch - self._finish_epoch >= 1000) and epoch >= 55000:
                print('finish epoch reached, quit')
                quit()


        return
    
    def save_run_val(self):

        config_json_path = os.path.join(self.cam_img_dir, 'config.json')
        val_config = json.load(open(config_json_path, 'r'))

        rewards_val = (self.ref_reward[0].clone() - 7).clamp(min=0).sum(dim=0)
        init_range_left_val_candidate = rewards_val.argmax()
        init_range_left_val = 0 if rewards_val[0] > rewards_val[init_range_left_val_candidate]-10 else init_range_left_val_candidate
        if not self.reverse_time:
            command = f'bash scripts/train_dual_forward_val.sh {val_config["seq_name"]} {val_config["gpu_id"]} {val_config["out_root"]} {val_config["motion_root"]} {val_config["cfg_env"]} {val_config["cfg_train"]} {self.curr_epoch} {init_range_left_val}'
        else:
            forward_data = os.path.join(self.cam_img_dir, 'ref_tar', f'ref_tar_{self.curr_epoch}','intermimic.pt').replace('backward','forward')
            while not os.path.exists(forward_data):
                time.sleep(2)
            time.sleep(10)
            command = f'bash scripts/train_dual_backward_val.sh {val_config["seq_name"]} {val_config["gpu_id"]} {val_config["out_root"]} {val_config["motion_root"]} {val_config["cfg_env"]} {val_config["cfg_train"]} {self.curr_epoch} {init_range_left_val}'
        os.system(command)
        
        print(f'run val with command: {command}')

        path_to_save_command = os.path.join(self.cam_img_dir, 'ref_tar', f'command_{self.curr_epoch}.txt')

        os.makedirs(os.path.dirname(path_to_save_command), exist_ok=True)



    
    def load_ref_traj(self):
        file_path = os.path.join(self.cam_img_dir, 'ref_traj',f'intermimic.pt')
        if self.reverse_time:
            file_path = file_path.replace('backward', 'forward')
        else:
            file_path = file_path.replace('forward', 'backward')
        if not os.path.exists(file_path):
            return
        finish_rate = self.ref_reward[0].max(dim=0).values/(torch.arange(self.max_episode_length[0], device=self.device).flip(0)+1e-6)
        if (finish_rate>0.85).sum()/self.max_episode_length[0] < 0.7:
            return

        highest_ref_reward_index = self.ref_reward[0].max(dim=0).values.argmax()
        left_of_highest_finish_rate = finish_rate[:highest_ref_reward_index]
        if left_of_highest_finish_rate[left_of_highest_finish_rate<0.55].numel() < left_of_highest_finish_rate.numel() *0.9:
            return
        if left_of_highest_finish_rate[left_of_highest_finish_rate<0.55].numel() < 5:
            return

        time.sleep(5)
        indices_of_low_finish_rate = torch.where(left_of_highest_finish_rate<0.55)[0]

        ref_traj_from_opposite = torch.load(file_path, map_location=self.device)
        if self.reverse_time:
            ref_traj_from_opposite = ref_traj_from_opposite.flip(0)

        valid_frames = ref_traj_from_opposite[:,-1]>0.5
        valid_frames[highest_ref_reward_index:]=False
        indices_of_valid_frames = torch.where(valid_frames)[0]
        if valid_frames.sum() < 5:
            return
        new_hoi_data = torch.zeros_like(self.hoi_data[0])
        new_hoi_data[indices_of_valid_frames] = self._load_new_motion(ref_traj_from_opposite[indices_of_valid_frames][:,:-1])

        mask_in_valid = torch.isin(indices_of_valid_frames, indices_of_low_finish_rate) 
        intersection = indices_of_valid_frames[mask_in_valid] 
        self.hoi_data[0,intersection] = new_hoi_data[intersection]
        self.hoi_data = self.hoi_data.contiguous()

        self.load_traj_cnt += 1
        return




    def load_ref_reward(self, epoch):
        if epoch >56000:
            quit()
        if not self.dual_update:
            return
        file_path = os.path.join(self.cam_img_dir, 'ref_reward',f'opposite_ref_reward_{epoch}.npz')
        file_path_for_hoi = os.path.join(self.cam_img_dir, 'ref_hoi',f'opposite_ref_hoi_{epoch}.npz')
        if self.reverse_time:
            file_path = file_path.replace('backward', 'forward')
            file_path_for_hoi = file_path_for_hoi.replace('backward', 'forward')

            while not (os.path.exists(file_path) and os.path.exists(file_path_for_hoi)):
                print(f'Waiting for {file_path} and {file_path_for_hoi} to be available...')
                time.sleep(2)  # Wait for 2 seconds before checking again

            time.sleep(5.5)
            loaded_dict = np.load(file_path, allow_pickle=True)
            ref_reward_from_opposite = torch.from_numpy(loaded_dict['ref_reward']).flip(2).to(self.device)
            loaded_dict = np.load(file_path_for_hoi, allow_pickle=True)
            hoi_refs_from_opposite = torch.from_numpy(loaded_dict['hoi_refs']).flip(2).to(self.device)
            indices_for_velocity = list(range(7, 13)) + list(range(166, 319)) + list(range(326, 332))
            hoi_refs_from_opposite[:,:, :, indices_for_velocity] *= -1

        else:
            file_path = file_path.replace('forward', 'backward')
            file_path_for_hoi = file_path_for_hoi.replace('forward', 'backward')
            while not (os.path.exists(file_path) and os.path.exists(file_path_for_hoi)):
                print(f'Waiting for {file_path} and {file_path_for_hoi} to be available...')
                time.sleep(2)  # Wait for 2 seconds before checking again
            time.sleep(5.5)
            loaded_dict = np.load(file_path, allow_pickle=True)
            ref_reward_from_opposite = torch.from_numpy(loaded_dict['ref_reward']).to(self.device)
            loaded_dict = np.load(file_path_for_hoi, allow_pickle=True)
            hoi_refs_from_opposite = torch.from_numpy(loaded_dict['hoi_refs']).to(self.device)

        value_opposite, index_opposite = ref_reward_from_opposite[:, 1:, :].max(dim=1)
        index_opposite = index_opposite + 1  # because we dropped the first frame
        value_current, index_current = self.ref_reward[:, 2:, :].min(dim=1)
        index_current = index_current + 2
        mask = (value_opposite > value_current + 10) & (value_opposite > 40) & (value_opposite > self.ref_reward[:, :2, :].max(dim=1)[0] * 5 / 4)
        self.ref_reward[:,index_current,:] = torch.where(mask, ref_reward_from_opposite[:,index_opposite,:]-10, self.ref_reward[:,index_current,:])

        self.hoi_refs[:,index_current,:] = torch.where(mask.unsqueeze(-1).expand(-1,-1,self.hoi_refs.shape[-1]), hoi_refs_from_opposite[:,index_opposite,:,:self.hoi_refs.shape[-1]], self.hoi_refs[:,index_current,:])
        self.contact_refs[:,index_current,:] = torch.where(mask.unsqueeze(-1).expand(-1,-1,self.contact_refs.shape[-1]), hoi_refs_from_opposite[:,index_opposite,:,self.hoi_refs.shape[-1]:], self.contact_refs[:,index_current,:])
        if epoch % 100 == 0 and epoch>=54300:
            self.load_ref_traj()
        if epoch % 100 == 0:
            self.load_run_val()
        return

    def get_longest_true_segment(self,mask):
        
        pad = torch.tensor([False], device=mask.device)
        diff = torch.diff(torch.cat([pad, mask, pad]).int())
        
        
        starts = torch.where(diff == 1)[0]
        ends = torch.where(diff == -1)[0]
        
        if starts.numel() == 0:
            return None, None, 0  
            
        
        idx = torch.argmax(ends - starts)
        
        
        return starts[idx].item(), ends[idx].item() - 1, (ends[idx] - starts[idx]).item()

    def load_run_val(self):
        self_path = os.path.join(self.cam_img_dir, 'ref_tar', f'ref_tar_{self.curr_epoch}/intermimic.pt')
        if not self.reverse_time:
            opposite_path = self_path.replace('forward','backward')
        else:
            opposite_path = self_path.replace('backward','forward')
            
        if not self.reverse_time:
            self_hoi_data_candidate = torch.load(self_path, map_location=self.device)
        else:
            self_hoi_data_candidate = torch.load(self_path, map_location=self.device).flip(0)
        self_indices = torch.where(self_hoi_data_candidate[:, -1] > 0.5)[0]
        if (self_indices.numel() >= self.max_episode_length[0]-1) and (self._finish_epoch>58500) and not self.reverse_time:
            self._finish_epoch = self.curr_epoch
                
        if self_indices.numel() > 0:
            self_indices = self_indices[:0] if torch.sum(self.contact_obj_whole[self_indices].clamp(0,1))/self_indices.numel() < 0.5 else self_indices
        if self_indices.numel() > 0 and self.curr_epoch >= 54400:
            left = self_indices[0].item()
            right = self_indices[-1].item()
            value_candidate = torch.zeros(self.max_episode_length, device=self.device)
            value_candidate[left:right+1] = torch.arange(right-left+1, device=self.device).flip(0)
            value_current, index_current = self.ref_reward[0, 1:, :].max(dim=0)
            index_current+=1
            mask = (value_candidate > value_current+30) & (value_candidate > value_current*3/2) & (value_candidate>60)
            mask_left, mask_right, mask_length = self.get_longest_true_segment(mask)

            
            path_to_save_command = os.path.join(self.cam_img_dir, 'ref_tar', f'info_self_{self.curr_epoch}.json')
            os.makedirs(os.path.dirname(path_to_save_command), exist_ok=True)
            to_save = {'left':left,'right':right,'mask_left': mask_left, 'mask_right': mask_right, 'mask_length': mask_length, 'value_candidate': value_candidate.detach().cpu().numpy().tolist(), 'value_current': value_current.detach().cpu().numpy().tolist(),'mask': mask.detach().cpu().numpy().tolist()}



                
            if mask_length > 30:
                mask=torch.zeros(self.max_episode_length ,dtype=torch.bool, device=self.device)
                mask[mask_left:mask_right+1] = True
                candidate_hoi_data = self._load_new_motion(self_path)
                self.hoi_data[0] = torch.where(mask.unsqueeze(-1).expand(-1,self.hoi_data.shape[-1]), candidate_hoi_data, self.hoi_data[0])

                    
                    
        while not os.path.exists(opposite_path):
            print(f'Waiting for {opposite_path} to be available...')
            time.sleep(2)  # Wait for 2 seconds before checking again
        time.sleep(10)
        if not self.reverse_time:
            opposite_hoi_data_candidate = torch.load(opposite_path, map_location=self.device)
        else:
            opposite_hoi_data_candidate = torch.load(opposite_path, map_location=self.device).flip(0)

        opposite_indices = torch.where(opposite_hoi_data_candidate[:, -1] > 0.5)[0]
        if (opposite_indices.numel() >= self.max_episode_length[0]-1) and (self._finish_epoch>58500) and self.reverse_time:
            self._finish_epoch = self.curr_epoch
            
        if opposite_indices.numel() > 0:
            opposite_indices = opposite_indices[:0] if torch.sum(self.contact_obj_whole[opposite_indices].clamp(0,1))/opposite_indices.numel() < 0.5 else opposite_indices
        if opposite_indices.numel() > 0 and self.curr_epoch >= 54400:
            left = opposite_indices[0].item()
            right = opposite_indices[-1].item()
            value_candidate = torch.zeros((self.max_episode_length,),device=self.device)
            value_candidate[left:right+1] = torch.arange(right-left+1,device=self.device).flip(0)
            value_current, index_current = self.ref_reward[0, 1:, :].min(dim=0)
            index_current+=1
            mask = (value_candidate > value_current+60) & (value_candidate > value_current*3/2) & (value_candidate>90)
            mask_left, mask_right, mask_length = self.get_longest_true_segment(mask)
            # print('oppo','mask_left, mask_right,mask_length',mask_left, mask_right,mask_length)
            
            path_to_save_command = os.path.join(self.cam_img_dir, 'ref_tar', f'info_oppo_{self.curr_epoch}.json')
            os.makedirs(os.path.dirname(path_to_save_command), exist_ok=True)
            to_save = {'left':left,'right':right,'mask_left': mask_left, 'mask_right': mask_right, 'mask_length': mask_length, 'value_candidate': value_candidate.detach().cpu().numpy().tolist(), 'value_current': value_current.detach().cpu().numpy().tolist(),'mask': mask.detach().cpu().numpy().tolist()}


                
                
            if mask_length > 30 or (mask_length>=5 and mask_left<=5):
                mask=torch.zeros(self.max_episode_length ,dtype=torch.bool, device=self.device)
                mask[mask_left:min(mask_right+11,self.max_episode_length)] = True
                candidate_hoi_data = self._load_new_motion(opposite_path)
                self.hoi_data[0] = torch.where(mask.unsqueeze(-1).expand(-1,self.hoi_data.shape[-1]), candidate_hoi_data, self.hoi_data[0])
                self._init_range_left_tar = max(0,mask_right-15)
                self._init_range_right_tar = max(1,mask_right-5)
                self._init_range_left_tar_0 = mask_left

                self.just_update_tar = True
            else:
                self.just_update_tar = False
        else:
            self.just_update_tar = False


        

    def update_play_hist(self, env_ids):


        mask = (self.progress_buf[env_ids] - self.start_times[env_ids]) > self.play_length_hist[self.start_times[env_ids]]
        self.play_length_hist[self.start_times[env_ids][mask]] = (self.progress_buf[env_ids][mask] - self.start_times[env_ids][mask])



        self.play_success_hist[self.start_times[env_ids]] = (self.progress_buf[env_ids] - self.start_times[env_ids])/ (self.max_episode_length[self.data_id[env_ids]] - self.start_times[env_ids])

        os.makedirs(self.cam_img_dir, exist_ok=True)
        if self.mode == 'test':
            if self.reverse_time:
                np.savez(os.path.join(self.cam_img_dir, 'play_length_hist_reverse.npz'), play_length_hist=self.play_length_hist.flip(0).cpu().numpy(), play_success_hist=self.play_success_hist.flip(0).cpu().numpy())
            else:
                np.savez(os.path.join(self.cam_img_dir, 'play_length_hist.npz'), play_length_hist=self.play_length_hist.cpu().numpy(), play_success_hist=self.play_success_hist.cpu().numpy())

            print('play_length_hist', torch.stack([self.play_length_hist, torch.arange(0, self.play_length_hist.shape[0], device=self.device)]).permute(1,0))


        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == InterMimic.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == InterMimic.StateInit.Start
              or self._state_init == InterMimic.StateInit.Random
              or self._state_init == InterMimic.StateInit.Traverse
              or self._state_init == InterMimic.StateInit.History
              or self._state_init == InterMimic.StateInit.Traverse_Random
              ):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == InterMimic.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        self._reset_target(env_ids)

        self._curr_state[env_ids, 0, :] = torch.cat([
            self._humanoid_root_states[env_ids],
            self._dof_pos[env_ids],
            self._dof_vel[env_ids],
            self._target_states[env_ids],
        ], dim=1)
        self._curr_state_complement[env_ids, 0, :] = torch.cat([
            self._rigid_body_pos[env_ids].reshape(env_ids.shape[0], -1),
            self._rigid_body_rot[env_ids].reshape(env_ids.shape[0], -1),
            self._rigid_body_vel[env_ids].reshape(env_ids.shape[0], -1),
            self._rigid_body_ang_vel[env_ids].reshape(env_ids.shape[0], -1),
        ], dim=1)
        return

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        i = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in env_ids], device=self.device, dtype=torch.long)

        if (self._state_init == InterMimic.StateInit.Random
            or self._state_init == InterMimic.StateInit.Hybrid):

            motion_times = torch.cat([torch.randint(self._init_range_left, self._init_range_right, (num_envs,), device=self.device, dtype=torch.long)])
        elif (self._state_init == InterMimic.StateInit.Start):

            motion_times = torch.ones(num_envs, device=self.device, dtype=torch.long) * self._init_range_left
        elif (self._state_init == InterMimic.StateInit.Traverse):

            motion_times = env_ids % (self._init_range_right - self._init_range_left) + self._init_range_left
        elif (self._state_init == InterMimic.StateInit.History):
            motion_times = torch.multinomial(self.init_prob_dist, num_envs, replacement=True) + self._init_range_left
        elif (self._state_init == InterMimic.StateInit.Traverse_Random):
            while self.Traverse_Random_Queue.shape[0] < num_envs:

                random_idx = torch.cat([torch.randperm(self._init_valid.size(0)) for _ in range(20)], dim=0).to(self.device)
                self.Traverse_Random_Queue = torch.cat([self.Traverse_Random_Queue, self._init_valid[random_idx]], dim=0)
            motion_times = self.Traverse_Random_Queue[:num_envs]
            self.Traverse_Random_Queue = self.Traverse_Random_Queue[num_envs:]


        ref_reward = self.ref_reward[i, :, motion_times] 
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        idx = 0
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0 
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )

        return

    def cal_cdf(self, i, e):
        """
        Calculate cumulative distribution function (CDF) for curriculum learning initialization.
        
        This function computes a probability distribution over time steps based on historical
        success rates, which is used to sample starting positions for new episodes.
        
        Args:
            i: Motion sequence indices for each environment
            e: Index of the specific environment to compute CDF for
            
        Returns:
            cdf: Cumulative distribution function over valid time steps [0, max_episode_length - rollout_length]
        """
        # Extract rewards for all reference trajectories at all valid time steps for this motion
        # Shape: [num_reference_trajectories, num_valid_timesteps]
        # Only consider timesteps that leave enough room for a full rollout
        # rewards = self.ref_reward[i[e], :, :max(1, self.max_episode_length[i[e]]-self.rollout_length)].clone() 
        rewards = (self.ref_reward[i[e], :,0 :self._init_range_right].clone() - 7).clamp(min=0)

        # Sum rewards across all reference trajectories for each time step
        # Then take reciprocal to convert rewards to "difficulty" (lower reward = higher difficulty)
        # Shape: [num_valid_timesteps]
        ref_reward_sum = rewards.sum(dim=0)

        finish_rate = (ref_reward_sum+14)/2 / (self.max_episode_length[0]-torch.arange(0, self._init_range_right).to(self.device))
        left_steps = (self.max_episode_length[0]-torch.arange(0, self._init_range_right).to(self.device)) - (ref_reward_sum+14)/2



        ref_reward_sum = torch.where(torch.logical_and(finish_rate>0.9, left_steps<=15), ref_reward_sum*0.2, ref_reward_sum)

        ref_reward_sum = torch.where(finish_rate>0.8, ref_reward_sum*0.5, ref_reward_sum)
        
        if self.just_update_tar:
            ref_reward_sum[self._init_range_left_tar: min(self._init_range_right_tar,self._init_range_right)] = (ref_reward_sum[self._init_range_left_tar: min(self._init_range_right_tar,self._init_range_right)]*1000).clamp_min(ref_reward_sum.max())
        
        if self.left_to_end_cnt > 100:
            ref_reward_sum[self._init_range_left]*=3





        # Normalize to create a probability distribution
        # Time steps with lower cumulative rewards (harder) get higher probability
        # This implements curriculum learning: start from harder positions as training progresses

        prob = ref_reward_sum / ref_reward_sum.sum()
        
        # Compute cumulative distribution function for sampling
        # Shape: [num_valid_timesteps]
        cdf = torch.cumsum(prob, 0)
        
        return cdf


    def _reset_hybrid_state_init(self, env_ids):
        # Get the number of environments to reset
        num_envs = env_ids.shape[0]
        
        # For each environment, randomly select a motion sequence that matches its object type
        # obj2motion is a boolean mask indicating which motions are compatible with each object
        i = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in env_ids], device=self.device, dtype=torch.long)
        
        # Create probability array for hybrid initialization (mix of reference and random starts)
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        
        # Randomly determine which environments will use reference initialization (True) vs random (False)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_init_mask = False
        # Get the IDs of environments that will use reference initialization
        ref_reset_ids = env_ids[ref_init_mask]

        # For each environment, determine the starting time in the motion sequence:
        # - If in ref_reset_ids: start from time 0 (beginning of motion)
        # - Otherwise: sample time from CDF based on historical success (curriculum learning)
        motion_times = torch.cat([torch.searchsorted(self.cal_cdf(i, e), torch.rand(1).to(self.device)) if env_ids[e] not in ref_reset_ids else torch.zeros((1,), device=self.device, dtype=torch.long) for e in range(num_envs)])
        
        # Get the reference rewards for the selected motions at the chosen time steps
        # Shape: [num_envs, num_reference_trajectories]
        # import pdb; pdb.set_trace()
        ref_reward = (self.ref_reward[i, :, motion_times] -6).clamp(min=1)
        
        # if self.just_update_tar:
        #     mask = (motion_times>=self._init_range_left_tar_0) & (motion_times<self._init_range_right_tar)
        #     ref_reward[2,mask]*=1000


        # Convert rewards to probabilities (normalize across reference trajectories)
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        # Compute cumulative distribution function for sampling reference trajectory
        cdf = torch.cumsum(prob, dim=1)
        
        # Sample which reference trajectory to use for each environment based on CDF
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        
        # Store the selected reference trajectory index for each environment
        self.ref_index[env_ids] = idx
        
        # Set the current progress and start time to the sampled motion time
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        
        # Store which motion sequence each environment is using
        self.data_id[env_ids] = i
        
        # Map motion ID to dataset ID
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        
        # Reset observation history and contact tracking
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0 
        
        # Set the actual physical state of the environment (positions, rotations, velocities)
        # by extracting the reference state at the chosen motion time and trajectory
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )
        return


    def _reset_hybrid_state_init_intermimic(self, env_ids):
        # Get the number of environments to reset
        num_envs = env_ids.shape[0]
        
        # For each environment, randomly select a motion sequence that matches its object type
        # obj2motion is a boolean mask indicating which motions are compatible with each object
        i = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in env_ids], device=self.device, dtype=torch.long)
        
        # Create probability array for hybrid initialization (mix of reference and random starts)
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        
        # Randomly determine which environments will use reference initialization (True) vs random (False)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        # Get the IDs of environments that will use reference initialization
        ref_reset_ids = env_ids[ref_init_mask]

        # For each environment, determine the starting time in the motion sequence:
        # - If in ref_reset_ids: start from time 0 (beginning of motion)
        # - Otherwise: sample time from CDF based on historical success (curriculum learning)
        motion_times = torch.cat([torch.searchsorted(self.cal_cdf(i, e), torch.rand(1).to(self.device)) if env_ids[e] not in ref_reset_ids else torch.zeros((1,), device=self.device, dtype=torch.long) for e in range(num_envs)]) 
        
        # Get the reference rewards for the selected motions at the chosen time steps
        # Shape: [num_envs, num_reference_trajectories]
        ref_reward = self.ref_reward[i, :, motion_times] 
        
        # Convert rewards to probabilities (normalize across reference trajectories)
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        # Compute cumulative distribution function for sampling reference trajectory
        cdf = torch.cumsum(prob, dim=1)
        
        # Sample which reference trajectory to use for each environment based on CDF
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        
        # Store the selected reference trajectory index for each environment
        self.ref_index[env_ids] = idx
        
        # Set the current progress and start time to the sampled motion time
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        
        # Store which motion sequence each environment is using
        self.data_id[env_ids] = i
        
        # Map motion ID to dataset ID
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        
        # Reset observation history and contact tracking
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0 
        
        # Set the actual physical state of the environment (positions, rotations, velocities)
        # by extracting the reference state at the chosen motion time and trajectory
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )
        return



    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _compute_task_obs(self, env_ids=None, ref_obs=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
        
        obs = self.compute_obj_observations(root_states, tar_states, ref_obs)
        return obs

    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids):
        # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h

        len_keypos = len(key_body_ids)
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                                heading_rot_expand_2.shape[2])
        
        heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_no_hand = heading_rot_expand.repeat((1, 22, 1))
        flat_heading_rot_no_hand = heading_rot_expand_no_hand.reshape(heading_rot_expand_no_hand.shape[0] * heading_rot_expand_no_hand.shape[1], 
                                                heading_rot_expand_no_hand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                heading_inv_rot_expand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand_no_hand = heading_inv_rot_expand.repeat((1, 22, 1))
        flat_heading_inv_rot_no_hand = heading_inv_rot_expand_no_hand.reshape(heading_inv_rot_expand_no_hand.shape[0] * heading_inv_rot_expand_no_hand.shape[1], 
                                                heading_inv_rot_expand_no_hand.shape[2])
        
        _ref_body_pos = self.extract_data_component('body_pos', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        _body_pos = body_pos[:, key_body_ids, :]

        diff_global_body_pos = _ref_body_pos - _body_pos
        diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
        
        local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)  # preserves the body position
        local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
    
        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = body_pos - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
        
        ref_body_rot = self.extract_data_component('body_rot', obs=ref_obs)
        ref_body_rot_no_hand = torch.cat((ref_body_rot[:, :18*4], ref_body_rot[:, 33*4:37*4]), dim=-1) 
        body_rot_no_hand = torch.cat((body_rot[:, :18], body_rot[:, 33:37]), dim=1)
        diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot_no_hand, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot_no_hand)
        diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot_no_hand, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        ref_body_vel = self.extract_data_component('body_pos_vel', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        _body_vel = body_vel[:, key_body_ids, :]
        diff_global_vel = ref_body_vel - _body_vel
        diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

        ref_body_ang_vel = self.extract_data_component('body_rot_vel', obs=ref_obs)
        ref_body_ang_vel_no_hand = torch.cat((ref_body_ang_vel[:, :18*3], ref_body_ang_vel[:, 33*3:37*3]), dim=-1)
        body_ang_vel_no_hand = torch.cat((body_ang_vel[:, :18], body_ang_vel[:, 33:37]), dim=1)
        diff_global_ang_vel = ref_body_ang_vel_no_hand.view(-1, 22, 3) - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot_no_hand, diff_global_ang_vel.view(-1, 3)).view(-1, 22 * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

        body_contact_buf = contact_forces[:, contact_body_ids, :].clone() #.view(contact_forces.shape[0],-1)
        contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
        ref_body_contact = self.extract_data_component('contact_human', obs=ref_obs)[:, contact_body_ids]
        diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)

        obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)
        return obs
    
    def compute_obj_observations(self, root_states, tar_states, ref_obs):
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]

        tar_pos = tar_states[:, 0:3]
        tar_rot = tar_states[:, 3:7]
        tar_vel = tar_states[:, 7:10]
        tar_ang_vel = tar_states[:, 10:13]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        local_tar_pos = tar_pos - root_pos
        local_tar_pos[..., -1] = tar_pos[..., -1]
        local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
        local_tar_vel = quat_rotate(heading_rot, tar_vel)
        local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

        local_tar_rot = quat_mul(heading_rot, tar_rot)
        local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

        _ref_obj_pos = self.extract_data_component('obj_pos', obs=ref_obs)
        diff_global_obj_pos = _ref_obj_pos - tar_pos
        diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

        local_ref_obj_pos = _ref_obj_pos - root_pos  # preserves the body position
        local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)

        ref_obj_rot = self.extract_data_component('obj_rot', obs=ref_obs)
        diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
        diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)  # Need to be change of basis
        diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

        local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
        local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)

        ref_obj_vel = self.extract_data_component('obj_pos_vel', obs=ref_obs)
        diff_global_vel = ref_obj_vel - tar_vel
        diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)

        ref_obj_ang_vel = self.extract_data_component('obj_rot_vel', obs=ref_obs)
        diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
        diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)

        obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)
        return obs
    
    def _compute_observations_iter(self, hoi_data, env_ids=None, delta_t=1):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)

        ts = self.progress_buf[env_ids].clone() 

        next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
        ref_obs = hoi_data[self.data_id[env_ids], next_ts].clone()
        obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts)
        task_obs = self._compute_task_obs(env_ids, ref_obs)
        obs = torch.cat([obs, task_obs], dim=-1)    
        ig_all, ig, ref_ig = self._compute_ig_obs(env_ids, ref_obs)
        return torch.cat((obs,ig_all,ref_ig-ig),dim=-1)
        
    def _compute_ig_obs(self, env_ids, ref_obs):
        ig = self.extract_data_component('ig', obs=self._curr_obs[env_ids]).view(env_ids.shape[0], -1, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
        ig_all = ig_all.view(env_ids.shape[0], -1)    
        ref_ig = self.extract_data_component('ig', obs=ref_obs)
        ref_ig = ref_ig.view(ref_obs.shape[0], -1, 3)[:, self._key_body_ids, :]
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
        ref_ig = ref_ig.view(env_ids.shape[0], -1)
        return ig_all, ig, ref_ig
        
    def _compute_observations(self, env_ids=None):
        if (env_ids is None):
            self._curr_ref_obs[:] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf[:] = torch.cat((self._compute_observations_iter(self.hoi_data, None, 1), self._compute_observations_iter(self.hoi_data, None, 16)), dim=-1)

        else:
            self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf[env_ids] = torch.cat((self._compute_observations_iter(self.hoi_data, env_ids, 1), self._compute_observations_iter(self.hoi_data, env_ids, 16)), dim=-1)
            
        return
    
    def _compute_hoi_observations(self, env_ids=None):
        self._curr_obs[:] = self.build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                        self._rigid_body_rot[:, 0, :],
                                                        self._rigid_body_vel[:, 0, :],
                                                        self._rigid_body_ang_vel[:, 0, :],
                                                        self._dof_pos, self._dof_vel, self._rigid_body_pos,
                                                        self._local_root_obs, self._root_height_obs, 
                                                        self._dof_obs_size, self._target_states,
                                                        self._tar_contact_forces,
                                                        self._contact_forces,
                                                        self.object_points[self.object_id[self.data_id]],
                                                        self._rigid_body_rot,
                                                        self._rigid_body_vel,
                                                        self._rigid_body_ang_vel
                                                        )
        return

    def build_hoi_observations(self, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, 
                            local_root_obs, root_height_obs, dof_obs_size, target_states, target_contact_buf, contact_buf, object_points, body_rot, body_vel, body_rot_vel):

        contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
        target_contact = torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)

        tar_pos = target_states[:, 0:3]
        tar_rot = target_states[:, 3:7]
        obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
        ig = compute_sdf(body_pos, obj_points).view(-1, 3)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, body_pos.shape[1], 1).view(-1, 4)
        ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1)    
        
        obs = torch.cat((root_pos, root_rot, dof_pos, dof_vel, 
                         body_pos.reshape(body_pos.shape[0],-1), body_rot.reshape(body_rot.shape[0],-1), body_vel.reshape(body_vel.shape[0],-1), body_rot_vel.reshape(body_rot_vel.shape[0],-1),
                         target_states, ig, contact, target_contact), dim=-1)
        return obs


    def _compute_reset(self):
        """
        Computes environment resets and updates reference motion database with successful trajectories.

        This function has two main responsibilities:
        1. Determine which environments need to be reset based on various failure conditions
        2. Update the reference motion database with successful trajectory segments for curriculum learning
        """

        # ===== PART 1: BASIC RESET COMPUTATION =====
        # Compute reset and termination flags for all environments
        # This calls the HOI-specific reset logic which extends humanoid reset conditions
        if self.mode == 'train':
            contact = torch.any(self.contact_reset[:,:] > 6, dim=-1)
        else:
            contact = torch.any(self.contact_reset[:,2:] > 10, dim=-1) and torch.any(self.contact_reset[:,:2] > 10, dim=-1)
        # contact = 0
        # contact = torch.any(self.contact_reset[:,2:] > 6, dim=-1)
        self.reset_buf[:], self._terminate_buf[:] = self.compute_hoi_reset(
            self.reset_buf, self.progress_buf, self.obs_buf,
            self._rigid_body_pos, self.max_episode_length[self.data_id],
            self._enable_early_termination, self._termination_heights, self.start_times,
            self.rollout_length, self.kinematic_reset, contact
        )
        self.finish_rate = (self.progress_buf - self.start_times) / (torch.min(torch.ones_like(self.progress_buf)*self.rollout_length, self.max_episode_length[self.data_id])-1)
        # ===== PART 2: CURRICULUM LEARNING - REFERENCE DATABASE UPDATE =====
        # Only proceed with database update if:
        # - Some environments are resetting (reset_buf.sum() > 0)
        # - Physical buffer size > 1 (self.psi > 1), indicating curriculum learning is enabled
        # import pdb; pdb.set_trace()
        
        if self.reset_buf.sum() > 0 and self.psi > 1:

            # Get indices of environments that are resetting
            reset_ind = (self.reset_buf == 1)
            data_id = self.data_id[reset_ind]  # Motion sequence IDs for resetting envs
            max_episode_length = self.max_episode_length[data_id]  # Max length for each motion



            # Get trajectory information for resetting environments
            start_index, end_index = self.start_times[reset_ind], self.progress_buf[reset_ind]
            self.to_end_cnt+=((end_index>=max_episode_length[0]-1) & (start_index<max_episode_length[0]-50)).sum().item()
            self.middle_to_end_cnt+= ((end_index>=max_episode_length[0]-1) & (start_index<max_episode_length[0]//2)).sum().item()
            self.left_to_end_cnt+= ((end_index>=max_episode_length[0]-1) & (start_index<=self._init_range_left+0)).sum().item()
            # print(self.to_end_cnt, self.middle_to_end_cnt,'middle_to_end_cnt',self.left_to_end_cnt)
            
            if self.left_to_end_cnt>200:
                self._init_range_left = 0
            

            curr_sum_reward = self._sum_reward[reset_ind] #.sum()  # Average reward (currently unused)
            curr_reward = self._curr_reward[reset_ind]  # Reward history for these episodes

            # Early exit condition (currently disabled: torch.rand(1)[0] < 0 is always False)
            # This could be used to randomly skip database updates

            # if torch.rand(1)[0] < 0:
            #     self._sum_reward[reset_ind] = 0
            #     return

            # Reset accumulated rewards for these environments
            self._sum_reward[reset_ind] = 0
            self._curr_reward[reset_ind] = 0
            # reset_ind = torch.logical_and(reset_ind, self.max_episode_length[self.data_id] > self.rollout_length)

            # Skip if too few environments meet the criteria (less than 99.5% threshold)
            # This ensures we have enough data for meaningful database updates
            if reset_ind.sum() < 0.995:
                return

            # Extract trajectory data from successful episodes
            
            state = self._curr_state[reset_ind]         # State history for these episodes
            state_complement = self._curr_state_complement[reset_ind]  # Complementary state history

            reward = torch.zeros((curr_reward.shape[0], self.hoi_refs.shape[0], self.hoi_refs.shape[2]), device=curr_reward.device)
            reward_opposite = torch.zeros((curr_reward.shape[0], self.hoi_refs.shape[0], self.hoi_refs.shape[2]), device=curr_reward.device)
            sum_reward = torch.zeros((curr_reward.shape[0], self.hoi_refs.shape[0], self.hoi_refs.shape[2]), device=curr_reward.device)
            
            for i in range(curr_reward.shape[0]):
                # Only process trajectories with sufficient length (30+ steps)
                # if end_index[i] > start_index[i] + 25:
                contact_obj = self.extract_data_component('contact_obj', obs=self.hoi_data[0, start_index[i]:end_index[i]])
                contact_obj_whole = self.extract_data_component('contact_obj', obs=self.hoi_data[0, 0:self.max_episode_length[0]])
                contact_obj_expand = self.extract_data_component('contact_obj', obs=self.hoi_data[0, max(0,start_index[i]-20):min(self.max_episode_length[0]-2,end_index[i]+20)])


                long_enough_1 = (end_index[i] - start_index[i] > 30) and ((torch.all(contact_obj_expand>0.1)) or (torch.all(contact_obj_whole<0.1)))
                long_enough_2 = (end_index[i] - start_index[i] > 70) and ((torch.sum(contact_obj)>70) or (torch.all(contact_obj_whole<0.1)))


                if long_enough_1 or long_enough_2:
                    if self.to_end_cnt>50 and end_index[i]>=max_episode_length[0]-1:
                        index_tensor = torch.arange(0, end_index[i]-start_index[i]+1, device=start_index.device).flip(0)
                        reward[i, data_id[i], start_index[i]:end_index[i]+1] = index_tensor
                        if end_index[i]-start_index[i] > 60:
                            reward_opposite[i, data_id[i], start_index[i]:end_index[i]+1] = index_tensor.flip(0)
                    else:
                        index_tensor = torch.arange(20, end_index[i]-start_index[i]+1, device=start_index.device).flip(0)
                        reward[i, data_id[i], start_index[i]:end_index[i]-20+1] = index_tensor
                        if end_index[i]-start_index[i] > 60:
                            reward_opposite[i, data_id[i], start_index[i]:end_index[i]-20+1] = index_tensor.flip(0)-10

                elif torch.all(contact_obj_expand>0.1):
                    reward[i, data_id[i], start_index[i]] = end_index[i] - start_index[i]

            adjust_reward, adjust_reward_index = reward.max(dim=0)
            adjust_reward_opposite, adjust_reward_index_opposite = reward_opposite.max(dim=0)
            # print(adjust_reward,adjust_reward_index)

            for i in range(reward.shape[1]):  # For each motion sequence
                for j in range(reward.shape[2]):  # For each time step
                    # Find the slot with minimum reward in the reference database (excluding slot 0)
                    # This is where we'll potentially insert the new successful trajectory
                    
                    value, index = self.ref_reward[i, 1:, j].min(dim=0)
                    index = index + 1  # Adjust for excluding slot 0

                    # Get the trajectory that achieved the best reward at this (motion, time)
                    id1 = adjust_reward_index[i, j]  # Which trajectory achieved best reward
                    idx = j - start_index[adjust_reward_index[i, j]]  # Relative time within that trajectory

                    if idx>=0:
                        sum_reward_to_be_compare = (curr_reward[id1, :end_index[adjust_reward_index[i, j]]-j+1] * self.powers[:end_index[adjust_reward_index[i, j]]-j+1]).sum()
                    else:
                        sum_reward_to_be_compare = 0

                    ratio = max((53250-self.curr_epoch)/1000,0)
                        
                    if idx >= 0 and ((adjust_reward[i, j] > value and sum_reward_to_be_compare >= self.ref_reward_sum[i, index, j]*ratio) or adjust_reward[i, j] > value + 10):
                        
                        # Replace the worst reference with this successful trajectory segment
                        self.ref_reward[i, index, j] = adjust_reward[i, j]  # Update reward
                        # import pdb; pdb.set_trace()
                        self.ref_reward_sum[i, index, j] = sum_reward_to_be_compare
                        # if idx > 0:
                        self.hoi_refs[i, index, j] = state[id1, idx]        # Update state reference
                        if idx > 0:
                            self.contact_refs[i, index, j] = self._curr_contact[reset_ind][id1, idx]

                    value_opposite, index_opposite = self.ref_reward_for_opposite[i, 1:, j].min(dim=0)
                    index_opposite = index_opposite + 1  # Adjust for excluding slot 0
                    id1_opposite = adjust_reward_index_opposite[i, j]  # Which trajectory achieved best reward
                    idx_opposite = j - start_index[adjust_reward_index_opposite[i, j]]  # Relative time within that trajectory
                    if idx_opposite >= 0 and adjust_reward_opposite[i, j] > value_opposite:
                        # Replace the worst reference with this successful trajectory segment
                        self.ref_reward_for_opposite[i, index_opposite, j] = adjust_reward_opposite[i, j]  # Update reward
                        # if idx_opposite > 0:
                        self.hoi_refs_for_opposite[i, index_opposite, j] = state[id1_opposite, idx_opposite]        # Update state reference

                        
        # self.ref_reward[:, 1:, :] = self.ref_reward[:, 1:, :] * (1 - 5e-4)
        if not self.just_update_tar:
            self.ref_reward[:, 1:, :] = self.ref_reward[:, 1:, :] * (1 - 5e-4)
            self.ref_reward_for_opposite[:, 1:, :] = self.ref_reward_for_opposite[:, 1:, :] * (1 - 5e-4)
            self.ref_reward_sum = self.ref_reward_sum * (1-5e-2)
        else:
            self.ref_reward[:, 1:, :] = self.ref_reward[:, 1:, :] * (1 - 5e-8)
            self.ref_reward_for_opposite[:, 1:, :] = self.ref_reward_for_opposite[:, 1:, :] * (1 - 5e-8)
            self.ref_reward_sum = self.ref_reward_sum * (1-5e-2)

        if True:
            if torch.sum(self.ref_reward>25)>3 and self.curr_epoch>30:
                # print('ref_reward>25', torch.sum(self.ref_reward>25))
                self._state_init = InterMimic.StateInit.Hybrid
            if self.curr_epoch>150 and torch.sum(self.ref_reward>12)>3:
                self._state_init = InterMimic.StateInit.Hybrid
        return
    

    def compute_hoi_reset(self, reset_buf, progress_buf, obs_buf, rigid_body_pos,
                          max_episode_length, enable_early_termination, termination_heights, 
                          start_times, rollout_length, reset_ig, contact_reset):

        reset, terminated = self.compute_humanoid_reset(reset_buf, progress_buf, obs_buf, rigid_body_pos,
                                                        max_episode_length, enable_early_termination, termination_heights, 
                                                        start_times, rollout_length)
        
        reset_ig *= (progress_buf > 1 + start_times)
        contact_reset *= (progress_buf > 1 + start_times)



        terminated = torch.where(torch.logical_or(reset_ig, contact_reset), torch.ones_like(reset_buf), terminated)
        reset = torch.where(reset.bool(), torch.ones_like(reset_buf), terminated)

        return reset, terminated

    def _compute_reward(self, actions):
        rb, human_reset, key_pos, ref_key_pos, all_pos, ref_all_pos = self.compute_humanoid_reward(self.reward_weights)
        ro, object_reset, obj_points, ref_obj_points = self.compute_obj_reward(self.reward_weights)
        rcc = self.compute_contact_chamfer_reward(self.reward_weights, all_pos, ref_all_pos, obj_points, ref_obj_points)
        rig, ig_reset = self.compute_ig_reward(self.reward_weights, key_pos, ref_key_pos, obj_points, ref_obj_points)
        rcg, contact_reset = self.compute_cg_reward(self.reward_weights)
        self.rew_buf[:] = rb * ro * rig * rcg * rcc
        kinematic_reset = torch.logical_or(human_reset, object_reset)
        if self.mode == 'train':
            self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        # self.contact_reset = (self.contact_reset + contact_reset*1.5 -0.5).clamp_min(0)
        else:
            self.contact_reset = (self.contact_reset + contact_reset)
        self.kinematic_reset = torch.logical_or(ig_reset, kinematic_reset)
        index = torch.arange(self._curr_reward.shape[0])

        self.reward_components = {
            'humanoid_reward': rb,
            'object_reward': ro,
            'interaction_graph_reward': rig,
            'contact_graph_reward': rcg,
            'contact_chamfer_reward': rcc,
        }


        try:
            if torch.isnan(self.rew_buf).any() or torch.isinf(self.rew_buf).any():
                print("NaN/inf detected in reward buffer")

            self._curr_reward[index, self.progress_buf - self.start_times] = self.rew_buf
            self._sum_reward[index] += self.rew_buf * 0.99
        except:
            print('error')

        self._curr_state[index, self.progress_buf - self.start_times, :] = torch.cat([
            self._humanoid_root_states, #0:13
            self._dof_pos, #13:166
            self._dof_vel, #166:319
            self._target_states, #319:332
        ], dim=1)

        self._curr_state_complement[index, self.progress_buf - self.start_times, :] = torch.cat([
            self._rigid_body_pos.reshape(self._rigid_body_pos.shape[0], -1),
            self._rigid_body_rot.reshape(self._rigid_body_rot.shape[0], -1),
            self._rigid_body_vel.reshape(self._rigid_body_vel.shape[0], -1),
            self._rigid_body_ang_vel.reshape(self._rigid_body_ang_vel.shape[0], -1),
        ], dim=1)

        
        
        left = self.start_times[0]
        right = self.max_episode_length.max()



        if self.mode == 'test' and ((torch.any(self.progress_buf[0] >= right-1) or torch.any(self.reset_buf[0] == 1))) and self.save_states:

            right = self.progress_buf[0]+1 if (torch.any(self.progress_buf[0] >= self.max_episode_length-1)) else max(self.progress_buf[0]-10, left+1)
            episode_state_all = self._curr_state[0, 0: right-left, :].cpu()
            episode_state_all_complement = self._curr_state_complement[0, 0:right-left, :].cpu()
            root_pos = episode_state_all[:, :3]
            root_rot = episode_state_all[:, 3:7]
            dof_pos = episode_state_all[:, 13:13+153]
            body_pos = episode_state_all_complement[:, :156]
            body_rot = episode_state_all_complement[:, 156:364]
            obj_pos = episode_state_all[:, 13+153+153:13+153+153+3]
            obj_rot = episode_state_all[:, 13+153+153+3:13+153+153+3+4]
            contact_object = self.extract_data_component('contact_obj', obs=self.hoi_data[0, left:right]).cpu()
            contact_human = self.extract_data_component('contact_human', obs=self.hoi_data[0, left:right]).cpu()
            new_hoi_data = torch.zeros((self.max_episode_length.max(), 591+1), dtype=self.hoi_data.dtype)

            new_hoi_data[left:right, :3] = root_pos
            new_hoi_data[left:right, 3:7] = root_rot
            new_hoi_data[left:right, 9:9+153] = dof_pos
            new_hoi_data[left:right, 162:162+156] = body_pos
            new_hoi_data[left:right, 318:321] = obj_pos
            new_hoi_data[left:right, 321:325] = obj_rot
            new_hoi_data[left:right, 330:331] = contact_object
            new_hoi_data[left:right, 331:331+52] = contact_human
            new_hoi_data[left:right, 331+52:331+52+52*4] = body_rot
            new_hoi_data[left:right, -1] = 1

            if self.reverse_time:
                new_hoi_data = new_hoi_data.flip(0)
            os.makedirs(self.cam_img_dir, exist_ok=True)
            torch.save(new_hoi_data, os.path.join(self.cam_img_dir, 'intermimic.pt'))
            quit()
        if self.mode == 'test' and torch.any(self.progress_buf[0] >= right-1):
            quit()
        return
    
    def compute_humanoid_reward(self, w):
        # body pos reward
        len_keypos = len(self._key_body_ids)
        

        all_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)
        key_pos = all_pos[:, self._key_body_ids]

        ref_all_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        ref_key_pos = ref_all_pos[:, self._key_body_ids]


        if self.reward_2d and w.get('p_2d', 0) > 0:
            key_2d,_ = self.project_points_to_camera(key_pos)
            ref_key_2d,_ = self.project_points_to_camera(ref_key_pos)
            ref_key_2d_pure = self.pure_2d_key[self.progress_buf-self.start_times][:, self._key_body_ids]  # Use current frame index to get corresponding 2D keypoints


        # Extract reference interaction graph and compute its norm
        ref_ig = self.extract_data_component('ig', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        ref_ig_norm = ref_ig.norm(dim=-1)
        
        # weight_h: Exponentially decreasing weight based on distance between body and object
        # Gives less importance to body parts that are far from the object
        weight_h = (-5 * ref_ig_norm).exp()
        
        # weight_hp: Modified version of weight_h used for position matching
        # Keeps the distance-based weights for most body parts but sets weight=1 for ankles and toes
        # This ensures consistent foot placement regardless of distance to object
        weight_hp = weight_h.clone().detach()  
        ancle_toe_ids = [i+1 for i in range(len_keypos) if 'Ankle' in self.key_bodies[i] or 'Toe' in self.key_bodies[i]]
        weight_hp[:, ancle_toe_ids] = 0.5

        # print(ref_key_pos.shape, key_pos.shape, weight_hp.shape, self._key_body_ids.shape)
        # Compute weighted position error between reference and current poses

        key_pos_with_root = key_pos.clone()
        root_pos = self.extract_data_component('root_pos', obs=self._curr_obs)
        key_pos_with_root[:,ancle_toe_ids] -= root_pos.unsqueeze(dim=1)
        ref_key_pos_with_root = ref_key_pos.clone()
        ref_root_pos = self.extract_data_component('root_pos', obs=self._curr_ref_obs)
        ref_key_pos_with_root[:,ancle_toe_ids] -= ref_root_pos.unsqueeze(dim=1)

        ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1) * weight_hp[:, self._key_body_ids],dim=-1)

        rp = torch.exp(-ep*w['p'])


        if self.reward_2d and w.get('p_2d', 0) > 0:
            e2d = torch.mean(((ref_key_2d_pure - key_2d)**2).sum(dim=-1),dim=-1)
            r2d = torch.exp(-e2d*w.get('p_2d', 0))
        else:
            e2d = torch.zeros_like(ep)
            r2d = torch.ones_like(rp)


        body_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)
        ref_body_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot.reshape(-1, 4))
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 52)
        weight_hr = 1 - weight_h
        
        er = torch.mean(diff[:, :] * weight_hr, dim=-1)
        rr = torch.exp(-er*w['r'])
        
        body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_obs)
        ref_body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_ref_obs)
        # body pos vel reward
        epv = torch.mean((ref_body_pos_vel - body_pos_vel)**2,dim=-1)
        # epv = torch.mean(pos_vel ,dim=-1) # torch.zeros_like(ep)
        rpv = torch.exp(-epv*w['pv'])

        dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_obs)
        ref_dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_ref_obs)
        # body rot vel reward
        erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2,dim=-1)
        rrv = torch.exp(-erv*w['rv'])

        # energy penalty
        hist_dof_vel = self.extract_data_component('dof_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('dof_vel', obs=self._curr_obs) - hist_dof_vel)*self.fps_data
        dof_diffacc = (local_vel.view(-1, 51*3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()
        energy = dof_diffacc.pow(2).mean(dim=-1).mul(-w['eg1']).exp()#.pow(max(1+(self.curr_epoch-53500)/2000,1))
        self.key_error = e2d
        rb = rp*rr*rpv*rrv*energy*r2d

        human_reset = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1) > 0.5
        
        return rb, human_reset, key_pos, ref_key_pos, all_pos, ref_all_pos
    
    def compute_obj_reward(self, w):

        root_pos = self.extract_data_component('root_pos', obs=self._curr_obs)
        root_rot = self.extract_data_component('root_rot', obs=self._curr_obs)

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

        if self.enable_camera_sensors and self.reward_2d and w.get('o_2d_maskany', 0) > 0:
            self.render_camera_sensors()

            obj_segs = (self.camera_seg_tensors==1).to(torch.float16)
            obj_segs_ref = self.ref_obj_mask[self.progress_buf]
   


            eo_2d_maskl2 = ((obj_segs_ref - obj_segs)**2).mean(dim=-1).mean(dim=-1)
            ro_2d_maskl2 = torch.exp(-eo_2d_maskl2*w.get('o_2d_maskl2', 0))

            if w.get('o_2d_maskot', 0) > 0:
                eo_2dmaskot = ImagesLoss("sinkhorn", p=2, blur=2.0, debias=True)(obj_segs_ref.float().unsqueeze(1)/(obj_segs_ref.float().unsqueeze(1).sum((2,3),keepdim=True)+1e-12), obj_segs.float().unsqueeze(1)/(obj_segs.float().unsqueeze(1).sum((2,3),keepdim=True)+1e-12))
                ro_2dmaskot = torch.exp(-eo_2dmaskot*w.get('o_2d_maskot', 0))
            else:
                ro_2dmaskot = torch.ones_like(ro_2d_maskl2)


            if w.get('o_2d_maskmiou', 0) > 0:
                if torch.sum(obj_segs_ref[0]) > 20:
                    eo_2d_maskmiou =  (obj_segs_ref * obj_segs) / (obj_segs_ref + obj_segs - (obj_segs_ref * obj_segs))
                else:
                    eo_2d_maskmiou = torch.ones_like(eo_2d_maskl2)*0.8
                ro_2d_maskmiou = torch.exp(eo_2d_maskmiou.mean(dim=-1).mean(dim=-1)*w.get('o_2d_maskmiou', 0))
            else:
                ro_2d_maskmiou = torch.ones_like(ro_2d_maskl2)

            if w.get('o_2d_maskcenter', 0) > 0:
                if torch.sum(obj_segs_ref[0]) > 10:
                    cx_ref = (obj_segs_ref * torch.arange(obj_segs_ref.size(-1), device=obj_segs_ref.device).view(1,1,-1)).sum(dim=(1,2)) / obj_segs_ref.sum(dim=(1,2)).clamp_min(1e-9)
                    cy_ref = (obj_segs_ref * torch.arange(obj_segs_ref.size(-2), device=obj_segs_ref.device).view(1,-1,1)).sum(dim=(1,2)) / obj_segs_ref.sum(dim=(1,2)).clamp_min(1e-9)
                    cx = (obj_segs * torch.arange(obj_segs.size(-1), device=obj_segs.device).view(1,1,-1)).sum(dim=(1,2)) / obj_segs.sum(dim=(1,2)).clamp_min(1e-9)
                    cy = (obj_segs * torch.arange(obj_segs.size(-2), device=obj_segs.device).view(1,-1,1)).sum(dim=(1,2)) / obj_segs.sum(dim=(1,2)).clamp_min(1e-9)
                    eo_2d_mask_center = ((cx_ref - cx)**2 + (cy_ref - cy)**2).sqrt()
                else:
                    eo_2d_mask_center = torch.ones_like(eo_2d_maskl2)*5
                ro_2d_mask_center = torch.exp(-eo_2d_mask_center*w.get('o_2d_maskcenter', 0))
            else:
                ro_2d_mask_center = torch.ones_like(ro_2d_maskl2)

        else:
            eo_2d_maskl2 = torch.zeros(self._curr_obs.shape[0], device=self.device)
            ro_2d_maskl2 = torch.ones_like(eo_2d_maskl2)
            ro_2d_maskmiou = ro_2d_maskl2
            ro_2d_mask_center = ro_2d_maskl2
            ro_2dmaskot = ro_2d_maskl2

        
        self.obj_seg_error = eo_2d_maskl2


        obj_pos = self.extract_data_component('obj_pos', obs=self._curr_obs)
        obj_rot = self.extract_data_component('obj_rot', obs=self._curr_obs)
        local_obj_pos = obj_pos - root_pos
        local_obj_pos[..., -1] = obj_pos[..., -1]
        local_obj_pos = quat_rotate(heading_rot, local_obj_pos)

        local_obj_rot = quat_mul(heading_rot, obj_rot)

        object_points = self.object_points[self.object_id[self.data_id]]
        obj_rot_extend = obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + obj_pos.unsqueeze(1)

        ref_root_pos = self.extract_data_component('root_pos', obs=self._curr_ref_obs)
        ref_root_rot = self.extract_data_component('root_rot', obs=self._curr_ref_obs)

        ref_heading_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)

        ref_obj_pos = self.extract_data_component('obj_pos', obs=self._curr_ref_obs)
        ref_obj_rot = self.extract_data_component('obj_rot', obs=self._curr_ref_obs)

        ref_local_obj_pos = ref_obj_pos - ref_root_pos
        ref_local_obj_pos[..., -1] = ref_obj_pos[..., -1]
        ref_local_obj_pos = quat_rotate(ref_heading_rot, ref_local_obj_pos)

        ref_local_obj_rot = quat_mul(ref_heading_rot, ref_obj_rot)

        ref_obj_rot_extend = ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        ref_obj_points = torch_utils.quat_rotate(ref_obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + ref_obj_pos.unsqueeze(1)

        eop = torch.mean(((ref_local_obj_pos - local_obj_pos)**2),dim=-1) # * (1 - weight_h.max(dim=-1)[0])
        rop = torch.exp(-eop*w['op'])

        # object rot reward
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_local_obj_rot), local_obj_rot)
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 1)
        
        eor = torch.mean(diff,dim=-1)
        ror = torch.exp(-eor*w['or'])

        obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_obs)
        ref_obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_ref_obs)
        # object pos vel reward
        eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
        ropv = torch.exp(-eopv*w['opv'])

        obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_obs)
        ref_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_ref_obs)
        # object rot vel reward
        eorv = torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
        rorv = torch.exp(-eorv*w['orv'])
        
        hist_obj_vel = self.extract_data_component('obj_pos_vel', obs=self._hist_obs)
        obj_diffacc = (self.extract_data_component('obj_pos_vel', obs=self._curr_obs) - hist_obj_vel)*self.fps_data
        obj_diffacc = obj_diffacc*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        hist_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('obj_rot_vel', obs=self._curr_obs) - hist_obj_rot_vel)*self.fps_data
        obj_rot_diffacc = local_vel.view(-1, 3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)
        
        obj_energy = (obj_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp()) * (obj_rot_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp())
        
        ro = rop*ror*ropv*rorv*obj_energy*ro_2d_maskl2*ro_2d_maskmiou*ro_2d_mask_center*ro_2dmaskot

        object_reset = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1) > 0.5
        return ro, object_reset, obj_points, ref_obj_points
    
    def compute_contact_chamfer_reward(self, w, all_pos, ref_all_pos, obj_points, ref_obj_points):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.left_hand_ids = list(range(17, 33))
        self.right_hand_ids = list(range(36, 52))

        left_hand_contact_any_ref = self.contact_label_left_hand[self.progress_buf]
        right_hand_contact_any_ref = self.contact_label_right_hand[self.progress_buf]
        hand_pos = all_pos[:,self.left_hand_ids+self.right_hand_ids]
        

        cdist = torch.cdist(hand_pos, obj_points)
        cdist_left = cdist[:,0:16]
        cdist_right = cdist[:,16:]

        contact_refs = self.contact_refs.view(self.contact_refs.shape[0], self.contact_refs.shape[1], self.contact_refs.shape[2], -1, 3)

        highest_ref_index = torch.argmax(self.ref_reward[0,:,self.progress_buf], dim=0)
        contact_valid_bit = contact_refs[0,highest_ref_index,self.progress_buf,0,0]>0.5

        cdist_left_new = torch.zeros(self.num_envs, 16, device=self.device)
        #get the minimum 3 distances on dim -1 of cdist_left
        self._curr_contact[env_ids,self.progress_buf-self.start_times,3:]= torch.cat([cdist_left.sort(dim=-1)[1][:,:,:3],cdist_right.sort(dim=-1)[1][:,:,:3]],dim=1).view(self.num_envs,-1).float()

        
        cdist_left_new[~contact_valid_bit] = cdist_left[~contact_valid_bit].min(dim=-1)[0]
        index_on_obj = contact_refs[0,highest_ref_index,self.progress_buf,1:17,:][contact_valid_bit].long() #排除第一位valid位
        cdist_left_new[contact_valid_bit] = torch.gather(cdist_left[contact_valid_bit], dim=-1, index=index_on_obj).mean(dim=-1)

        cdist_right_new = torch.zeros(self.num_envs, 16, device=self.device)
        cdist_right_new[~contact_valid_bit] = cdist_right[~contact_valid_bit].min(dim=-1)[0]
        index_on_obj = contact_refs[0,highest_ref_index,self.progress_buf,17:,:][contact_valid_bit].long()

        cdist_right_new[contact_valid_bit] = torch.gather(cdist_right[contact_valid_bit], dim=-1, index=index_on_obj).mean(dim=-1)



        cc_left_error = torch.zeros(self.num_envs, device=self.device)
        cc_right_error = torch.zeros(self.num_envs, device=self.device)


        cc_left_error[~left_hand_contact_any_ref] =0.05
        cc_right_error[~right_hand_contact_any_ref] =0.05
        cc_left_error[left_hand_contact_any_ref] = cdist_left_new[left_hand_contact_any_ref].mean(dim=-1)
        cc_right_error[right_hand_contact_any_ref] = cdist_right_new[right_hand_contact_any_ref].mean(dim=-1)
        cc_reward = torch.exp(-w['cc'] * (cc_left_error + cc_right_error)) *3
        return cc_reward

    


    def compute_ig_reward(self, w, key_pos, ref_key_pos, obj_points, ref_obj_points):
        len_keypos = len(self._key_body_ids)
        ig = key_pos.view(-1,len_keypos,3).unsqueeze(2) - obj_points.unsqueeze(1)
        ref_ig = ref_key_pos.view(-1,len_keypos,3).unsqueeze(2) - ref_obj_points.unsqueeze(1)
        ### interaction graph reward ###
        weight_1 = (1 / torch.clamp((ig**2).sum(dim=-1), min=0.01))
        weight_1 = weight_1 / weight_1.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        weight_2 = (1 / torch.clamp((ref_ig**2).sum(dim=-1), min=0.01))
        weight_2 = weight_2 / weight_2.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        eig = ((ig - ref_ig)**2).sum(dim=-1) * (weight_1 + weight_2)  
        if w.get('ig', 0) > 0:
            rig = torch.exp(-w['ig'] * (eig.sum(dim=-1).sum(dim=-1) * 0.5))
        else:
            rig = torch.ones_like(eig.sum(dim=-1).sum(dim=-1)) * 0.1

        reset_ig_1 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ref_ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig_2 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)
        reset_ig = torch.zeros_like(reset_ig)
        return rig, reset_ig
    
    def compute_cg_reward(self, w):    
        contact_thres = 0.1
        ref_human_contact = self.extract_data_component('contact_human', obs=self._curr_ref_obs)
        human_contact = self.extract_data_component('contact_human', obs=self._curr_obs)
        left_contact_hand_ids = list(range(17, 33))

        
        ref_left_contact_hand = ref_human_contact[:, left_contact_hand_ids]
        ref_left_contact_hand_any = torch.any(ref_left_contact_hand > contact_thres, dim=-1).float()
        left_hand_contact = human_contact[:, left_contact_hand_ids].clone()
        left_hand_contact_any_finger = torch.any(left_hand_contact[:,1:13] > contact_thres, dim=-1, keepdim=True).float()
        left_hand_contact_any_palm = torch.any(left_hand_contact[:,:1] > contact_thres, dim=-1, keepdim=True).float()

        ecg_left = (((ref_left_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(left_hand_contact - ref_left_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_left = 0.5 * (1 + torch.exp(-ecg_left*w['cg_hand'])) * (ref_left_contact_hand_any) + (1 - ref_left_contact_hand_any)


        right_contact_hand_ids = list(range(36, 52))
        
        ref_right_contact_hand = ref_human_contact[:, right_contact_hand_ids]
        ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()
        right_hand_contact = human_contact[:, right_contact_hand_ids].clone()
        right_hand_contact_any_finger = torch.any(right_hand_contact[:,1:13] > contact_thres, dim=-1, keepdim=True).float()
        right_hand_contact_any_palm = torch.any(right_hand_contact[:,:1] > contact_thres, dim=-1, keepdim=True).float()

        contact_reset = torch.cat([ 
                                torch.abs(ref_left_contact_hand_any.unsqueeze(-1) - left_hand_contact_any_finger) * ref_left_contact_hand_any.unsqueeze(-1), 
                                torch.abs(ref_right_contact_hand_any.unsqueeze(-1) - right_hand_contact_any_finger) * ref_right_contact_hand_any.unsqueeze(-1),
                                torch.abs(ref_left_contact_hand_any.unsqueeze(-1) - left_hand_contact_any_palm) * ref_left_contact_hand_any.unsqueeze(-1), 
                                torch.abs(ref_right_contact_hand_any.unsqueeze(-1) - right_hand_contact_any_palm) * ref_right_contact_hand_any.unsqueeze(-1),
                                ], dim=-1)


        ecg_right = (((ref_right_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(right_hand_contact - ref_right_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_right = 0.5 * (1 + torch.exp(-ecg_right*w['cg_hand'])) * (ref_right_contact_hand_any) + (1 - ref_right_contact_hand_any)
        
        rcg_hand = rcg_left * rcg_right

        other_ids = [i for i in range(len(self.contact_bodies)) if i not in left_contact_hand_ids and i not in right_contact_hand_ids]
        ref_other_contact = ref_human_contact[:, other_ids]
        other_contact = human_contact[:, other_ids]
        ecg_other = ((torch.abs(other_contact - ref_other_contact) * (ref_other_contact > contact_thres))).mean(dim=-1)
        rcg_other = torch.exp(-ecg_other*w['cg_other'])
        
        no_contact = torch.abs(human_contact) < contact_thres
        ecg_all = (torch.abs(no_contact + ref_human_contact) * (ref_human_contact < -contact_thres)).mean(dim=-1)
        rcg_all = torch.exp(-ecg_all*w['cg_all'])

        contact_all = self._contact_forces.clone().abs().sum(dim=-1).sum(dim=-1)
        contact_energy = contact_all.pow(2).mul(-w['eg3']).exp()

        rcg = rcg_hand*rcg_other*rcg_all*contact_energy
        return rcg, contact_reset
    
    def play_dataset_step(self, time):

        t = time
        if t == 0:
            self.data_id = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        env_ids = to_torch([i for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        t = to_torch(
                [
                    t if t < self.max_episode_length[self.data_id[i]] else self.max_episode_length[self.data_id[i]]-1
                    for i in range(self.num_envs)
                ],
                device=self.device,
                dtype=torch.long
            )
        ### update object ###
        obj_pos = self.extract_data_component('obj_pos', True, self.data_id[env_ids], t)
        obj_rot = self.extract_data_component('obj_rot', True, self.data_id[env_ids], t)
        self._target_states[env_ids, :3] = obj_pos
        self._target_states[env_ids, 3:7] = obj_rot
        self._target_states[env_ids, 7:13] = 0.0

        ### update subject ###   
        _humanoid_root_pos = self.extract_data_component('root_pos', True, self.data_id[env_ids], t)
        _humanoid_root_rot = self.extract_data_component('root_rot', True, self.data_id[env_ids], t)
        self._humanoid_root_states[env_ids, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[env_ids, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[env_ids, 7:13] = 0.0
        
        self._dof_pos[env_ids] = self.extract_data_component('dof_pos', True, self.data_id[env_ids], t)
        self._dof_vel[env_ids] = 0.0

        body_pos = self.extract_data_component('body_pos', True, self.data_id[env_ids], t).view(len(env_ids), self.num_bodies, 3)
        body_rot = self.extract_data_component('body_rot', True, self.data_id[env_ids], t).view(len(env_ids), self.num_bodies, 4)
        self._body_proxy_states[env_ids, :, 0:3] = body_pos
        self._body_proxy_states[env_ids, :, 3:7] = body_rot
        self._body_proxy_states[env_ids, :, 7:13] = 0.0

        actor_ids = torch.cat((
            self._humanoid_actor_ids[env_ids],
            self._tar_actor_ids[env_ids],
            self._body_proxy_actor_ids[env_ids].reshape(-1),
        ), dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        humanoid_actor_ids = self._humanoid_actor_ids[env_ids]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(humanoid_actor_ids), len(humanoid_actor_ids))

        obj_contact = self.extract_data_component('contact_obj', True, self.data_id[env_ids], t)
        obj_contact = torch.any(obj_contact > 0.1, dim=-1)
        human_contact = self.extract_data_component('contact_human', True, self.data_id[env_ids], t)
        for env_id, env_ptr in enumerate(self.envs):
            if env_id in env_ids:
                env_ptr = self.envs[env_id]
                handle = self._target_handles[env_id]

                if obj_contact[env_id] == True:
                    self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(1., 0., 0.))
                else:
                    self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(0., 0., 1.))
                    
                body_proxy_handles = self._body_proxy_handles[env_id]
                handle = self.humanoid_handles[env_id]
                for j in range(self.num_bodies):
                    color_handle = body_proxy_handles[j]
                    if human_contact[env_id, j] > 0.5:
                        self.gym.set_rigid_body_color(env_ptr, color_handle, 0, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(1., 0., 0.))
                    elif human_contact[env_id, j] > -0.5:
                        self.gym.set_rigid_body_color(env_ptr, color_handle, 0, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 1., 0.))
                    else:
                        self.gym.set_rigid_body_color(env_ptr, color_handle, 0, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 0., 1.))
        self.render(t=t)

        if self.enable_camera_sensors:
            if not hasattr(self, 't_before'):
                self.t_before = 0
            self.render_camera_sensors()
            if self.save_images:
                self.save_cam_imgs(t=t)
                self.save_cam_segs(t=t)
                self.save_2d_keypoints(t=t)
            self.t_before+=1

        return
    
    def save_cam_imgs(self, t=torch.tensor([0])):
        env_ids = 0
        img = self.camera_image_tensors.detach().cpu().numpy()[env_ids].astype(np.uint8)
        img = img.reshape(img.shape[0], -1, 4)[:, :, :3]
        if self.play_dataset:
            frame_id = t[env_ids]
        else:
            frame_id = self.progress_buf[env_ids]
        dataname = '_'.join(self.motion_file[-1].split('/')[-2:])
        rgb_filename = self.cam_img_dir + "/" + 'camera_images' + "/%05d.png" % (frame_id)
        os.makedirs(self.cam_img_dir + "/" + 'camera_images', exist_ok=True)
        cv2.imwrite(rgb_filename, img)
        return


    def save_cam_segs(self, t=torch.tensor([0])):
        if not hasattr(self, 't_before'):
            self.t_before = 0
        env_ids = 0
        img = (self.camera_seg_tensors.detach().cpu().numpy()[env_ids]==1).astype(np.uint8)*255
        if self.play_dataset:
            frame_id = t[env_ids]
        else:
            frame_id = self.progress_buf[env_ids]

        dataname = '_'.join(self.motion_file[-1].split('/')[-2:])
        rgb_filename = self.cam_img_dir + "/" + 'seg_images' + "/%05d.png" % (frame_id)
        os.makedirs(self.cam_img_dir + "/" + 'seg_images', exist_ok=True)
        cv2.imwrite(rgb_filename, img)
        return
    
    def save_2d_keypoints(self, t=torch.tensor([0])):
        env_ids = 0
        if self.play_dataset:
            frame_id = t[env_ids]
        else:
            frame_id = self.progress_buf[env_ids]
        # Extract body positions and check if they have been mean-centered
        if self.play_dataset:
            # root_pos = self.extract_data_component('root_pos', True, self.data_id[env_ids], t)
            body_pos = self.extract_data_component('body_pos', True, self.data_id[env_ids], t)[0]
        else:
            # root_pos = self.extract_data_component('root_pos', obs=self._curr_obs).
            body_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1)[0]
        # Looking at the code context, this body_pos appears to be raw values 
        # without any mean subtraction, as it's directly used for projection
        key_2d, key_pixel = self.project_points_to_camera(body_pos.view(1, -1, 3))
        key_2d = key_2d.view(-1, 2).detach().cpu()
        key_pixel = key_pixel.view(-1, 2)[:,:].detach().cpu().numpy()

        key_filename = self.cam_img_dir + "/" + 'keypt' + "/keypt_env%d_frame%05d.pt" % (env_ids, frame_id)
        os.makedirs(self.cam_img_dir + "/" + 'keypt', exist_ok=True)
        torch.save(key_2d, key_filename)

        width = 400
        height = 224
        img = np.zeros((height, width, 4), dtype=np.uint8)
        for i in range(key_pixel.shape[0]):
            x, y = key_pixel[i]
            x, y = int(x), int(y)
            if x >= 0 and x < width and y >= 0 and y < height:
                # Draw a filled circle (ball) for each keypoint
                radius = 0  # Radius of the ball
                cv2.circle(img, (x, 224 -y), radius, (255, 255, 255, 255), -1)  # -1 means filled circle

        dataname = '_'.join(self.motion_file[-1].split('/')[-2:])
        rgb_filename = self.cam_img_dir + "/" + 'key_images' + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
        os.makedirs(self.cam_img_dir + "/" + 'key_images', exist_ok=True)
        cv2.imwrite(rgb_filename, img)
        return


    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)

        if self.viewer:  
            if self.save_images:
                env_ids = 0
                if self.play_dataset:
                    frame_id = t
                else:
                    frame_id = self.progress_buf[env_ids]
                dataname = self.motion_file[-1][6:-3]
                rgb_filename = "intermimic/data/images/" + dataname + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                os.makedirs("intermimic/data/images/" + dataname, exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer,rgb_filename)
        return
    
            

@torch.jit.script
# This function computes the signed distance field (SDF) between two point clouds
def compute_sdf(points1, points2):
    # points1: [batch_size, num_points1, 3] - First set of points (e.g., body points)
    # points2: [batch_size, num_points2, 3] - Second set of points (e.g., object points)
    # Returns: [batch_size, num_points1, 3] - Vector from each point in points1 to nearest point in points2
    
    # Compute pairwise distances between all points
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)  # [batch, num_points1, num_points2, 3]
    
    # Calculate Euclidean distances
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)  # [batch, num_points1, num_points2]
    
    # Find indices of minimum distances for each point in points1
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)  # [batch, num_points1]
    
    # Create indices for batch and point dimensions
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    
    # Get vectors to nearest points
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat
