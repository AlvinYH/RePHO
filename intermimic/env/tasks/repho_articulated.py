"""RePHO Studio task for the shared passive articulated scene."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import to_torch
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from env.tasks.intermimic import InterMimic, compute_sdf
from utils import torch_utils


def _link_poses_in_object_frame(reference):
    """Express reference link poses in the reference object-root frame."""

    root_pos = np.asarray(reference["root_pos"])
    root_rot = np.asarray(reference["root_rot"])
    link_pos = np.asarray(reference["link_pos"])
    link_rot = np.asarray(reference["link_rot"])
    frames = root_pos.shape[0]
    if root_pos.shape != (frames, 3) or root_rot.shape != (frames, 4):
        raise ValueError("Object root reference must have shape (frames, 3/4)")
    if link_pos.ndim != 3 or link_pos.shape != (frames, link_pos.shape[1], 3):
        raise ValueError("Object link positions must have shape (frames, links, 3)")
    if link_rot.shape != (frames, link_pos.shape[1], 4):
        raise ValueError("Object link rotations must have shape (frames, links, 4)")
    if not all(np.isfinite(value).all() for value in (root_pos, root_rot, link_pos, link_rot)):
        raise ValueError("Object root and link references must be finite")
    if link_pos.shape[1] == 0:
        return {"pos": link_pos.astype(np.float32, copy=True), "rot": link_rot.astype(np.float32, copy=True)}

    links = Rotation.from_quat(link_rot.reshape(-1, 4))
    root_per_link = Rotation.from_quat(np.repeat(root_rot, link_pos.shape[1], axis=0))
    return {
        "pos": root_per_link.inv().apply(
            (link_pos - root_pos[:, None]).reshape(-1, 3)
        ).reshape(link_pos.shape).astype(np.float32),
        "rot": (root_per_link.inv() * links).as_quat().reshape(link_rot.shape).astype(np.float32),
    }


def _place_links_at_object_root(root, local_links):
    """Place root-local link poses at the current repaired object root."""

    root_pos = root["pos"]
    root_rot = root["rot"]
    local_pos = local_links["pos"]
    local_rot = local_links["rot"]
    batch = root_pos.shape[0]
    if root_pos.shape != (batch, 3) or root_rot.shape != (batch, 4):
        raise ValueError("Object root tensors must have shape (batch, 3/4)")
    if local_pos.ndim != 3 or local_pos.shape != (batch, local_pos.shape[1], 3):
        raise ValueError("Local link positions must have shape (batch, links, 3)")
    if local_rot.shape != (batch, local_pos.shape[1], 4):
        raise ValueError("Local link rotations must have shape (batch, links, 4)")

    root_xyz = root_rot[:, None, :3]
    root_w = root_rot[:, None, 3:4]
    cross = torch.linalg.cross(root_xyz.expand_as(local_pos), local_pos, dim=-1)
    rotated = local_pos + 2.0 * torch.linalg.cross(
        root_xyz.expand_as(local_pos), cross + root_w * local_pos, dim=-1
    )
    local_xyz = local_rot[..., :3]
    local_w = local_rot[..., 3:4]
    world_xyz = (
        root_w * local_xyz
        + local_w * root_xyz
        + torch.linalg.cross(root_xyz.expand_as(local_xyz), local_xyz, dim=-1)
    )
    world_w = root_w * local_w - torch.sum(root_xyz * local_xyz, dim=-1, keepdim=True)
    world_rot = torch.cat((world_xyz, world_w), dim=-1)
    return {
        "pos": rotated + root_pos[:, None],
        "rot": world_rot / torch.linalg.norm(world_rot, dim=-1, keepdim=True),
    }


def _mean_normalized_joint_error(joint):
    return torch.mean(
        ((joint["qpos"] - joint["reference"]) / joint["range"]) ** 2,
        dim=1,
    )


def _load_humanoid_tree(path):
    root = ET.parse(path).getroot().find("worldbody/body")
    if root is None:
        raise ValueError(f"Humanoid MJCF has no worldbody root: {path}")
    parents = []
    offsets = []

    def add(body, parent):
        index = len(parents)
        parents.append(parent)
        offsets.append(np.fromstring(body.get("pos", "0 0 0"), sep=" "))
        for child in body.findall("body"):
            add(child, index)

    add(root, -1)
    return parents, np.asarray(offsets, dtype=np.float32)


def _object_creation_pose(root_pos, root_rot, reverse_time):
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"object_root_pos must have shape (frames, 3), got {root_pos.shape}")
    if root_rot.shape != (root_pos.shape[0], 4):
        raise ValueError(
            "object_root_rot_xyzw must have shape (frames, 4), got "
            f"{root_rot.shape}"
        )
    if not np.all(np.isfinite(root_pos)) or not np.all(np.isfinite(root_rot)):
        raise ValueError("Object root reference must be finite")
    norm_error = np.max(np.abs(np.linalg.norm(root_rot, axis=1) - 1.0))
    if norm_error > 1e-4:
        raise ValueError(f"Object root quaternion norm error is too large: {norm_error}")
    frame = -1 if reverse_time else 0
    return root_pos[frame].copy(), root_rot[frame].copy()


def _creation_dof_state(state, initial_qpos, initial_qvel):
    if state.shape != (len(initial_qpos),):
        raise ValueError(
            "Actor DOF state and reference-frame-0 qpos disagree: "
            f"{state.shape} vs {initial_qpos.shape}"
        )
    if initial_qvel.shape != initial_qpos.shape:
        raise ValueError("Reference-frame-0 qpos and qvel disagree")
    if state.dtype.names is None or not {"pos", "vel"}.issubset(state.dtype.names):
        raise ValueError("Actor DOF state must expose pos/vel fields")
    state = state.copy()
    state["pos"] = initial_qpos
    state["vel"] = initial_qvel
    return state


class RePHOArticulated(InterMimic):
    """Preserve RePHO's native repair loop while adding q/link task state."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        global box_region_surface_distances
        global capsule_region_surface_distances
        global hand2_body_groups
        global load_mjcf_body_boxes
        global load_mjcf_body_capsules
        from pipeline.physics.common_rollout import CommonRolloutRecorder, SMPLX_BODY_NAMES
        from pipeline.physics.contact import (
            box_region_surface_distances,
            capsule_region_surface_distances,
            hand2_body_groups,
            load_mjcf_body_boxes,
            load_mjcf_body_capsules,
        )
        from pipeline.physics.articulated_scene import (
            ARTICULATED_OBJECT_COLLISION_FILTER,
            STATIC_SCENE_COLLISION_FILTER,
            add_ground_plane,
            configure_articulated_actor,
            create_static_box_actors,
            load_articulated_asset,
            load_static_box_assets,
            validate_humanoid_object_collision_filters,
        )

        env = cfg["env"]
        self._object_collision_filter = ARTICULATED_OBJECT_COLLISION_FILTER
        self._static_collision_filter = STATIC_SCENE_COLLISION_FILTER
        self._add_ground_plane = add_ground_plane
        self._validate_humanoid_object_collision_filters = (
            validate_humanoid_object_collision_filters
        )
        manifest_path = Path(env["articulatedInputPath"]).expanduser().resolve()
        self._art_config = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(
            Path(self._art_config["reference_path"]).expanduser().resolve(),
            allow_pickle=False,
        ) as values:
            object_root_pos = np.asarray(values["object_root_pos"], dtype=np.float32)
            object_root_rot = np.asarray(values["object_root_rot_xyzw"], dtype=np.float32)
            self._art_qref_np = np.asarray(values["object_joint_qpos"], dtype=np.float32)
            self._art_joint_types = [
                str(value) for value in np.asarray(values["joint_types"]).tolist()
            ]
            reference_joint_names = [
                str(value) for value in np.asarray(values["joint_names"]).tolist()
            ]
            reference_active_joint_names = [
                str(value)
                for value in np.asarray(values["active_joint_names"]).tolist()
            ]
            reference_active_parent_names = [
                str(value)
                for value in np.asarray(values["active_parent_link_names"]).tolist()
            ]
            reference_active_child_names = [
                str(value)
                for value in np.asarray(values["active_child_link_names"]).tolist()
            ]
            self._art_link_ref_np = np.asarray(values["object_link_pos"], dtype=np.float32)
            self._art_link_ref_rot_np = np.asarray(
                values["object_link_rot_xyzw"], dtype=np.float32
            )
            self._art_link_names = [
                str(value) for value in np.asarray(values["body_names"]).tolist()
            ]
            self._art_intended_np = np.asarray(values["intended"], dtype=np.bool_)
            self._art_contact_points_np = np.asarray(
                values["contact_points_link_local_scaled"], dtype=np.float32
            )
            self._art_contact_point_links = [
                str(value)
                for value in np.asarray(values["contact_point_link_names"]).tolist()
            ]
            self._art_contact_region_link_names = [
                str(value)
                for value in np.asarray(values["contact_region_link_names"]).tolist()
            ]
            self._art_reference_fps = float(np.asarray(values["fps"]).item())
            parent_points = np.asarray(values["active_parent_points"], dtype=np.float32)
            child_points = np.asarray(values["active_child_points"], dtype=np.float32)
        if bool(env.get("reverse_time", False)):
            object_root_pos = object_root_pos[::-1].copy()
            object_root_rot = object_root_rot[::-1].copy()
            self._art_qref_np = self._art_qref_np[::-1].copy()
            self._art_link_ref_np = self._art_link_ref_np[::-1].copy()
            self._art_link_ref_rot_np = self._art_link_ref_rot_np[::-1].copy()
            self._art_intended_np = self._art_intended_np[::-1].copy()
        self._object_creation_pos_np, self._object_creation_rot_np = _object_creation_pose(
            object_root_pos,
            object_root_rot,
            False,
        )
        local_links = _link_poses_in_object_frame(
            {
                "root_pos": object_root_pos,
                "root_rot": object_root_rot,
                "link_pos": self._art_link_ref_np,
                "link_rot": self._art_link_ref_rot_np,
            }
        )
        self._art_link_local_np = local_links["pos"]
        self._art_link_local_rot_np = local_links["rot"]
        self._art_joint_names = [str(value) for value in self._art_config["joint_names"]]
        self._art_active_joint_names = [
            str(value) for value in self._art_config["active_joint_names"]
        ]
        self._art_active_dof_ids_np = np.asarray(
            [
                self._art_joint_names.index(name)
                for name in self._art_active_joint_names
            ],
            dtype=np.int64,
        )
        self._art_q0_np = self._art_qref_np[0].copy()
        self._art_qvel0_np = (
            self._art_qref_np[1] - self._art_qref_np[0]
        ) * self._art_reference_fps
        self._art_dof_count = len(self._art_joint_names)
        self._art_active_dof_count = len(self._art_active_joint_names)
        self._art_active_link_names = [
            str(value)
            for value in self._art_config["active_child_link_names"]
        ]
        if (
            reference_joint_names != self._art_joint_names
            or self._art_joint_types
            != [str(value) for value in self._art_config["joint_types"]]
            or reference_active_joint_names != self._art_active_joint_names
            or reference_active_parent_names
            != [
                str(value)
                for value in self._art_config["active_parent_link_names"]
            ]
            or reference_active_child_names != self._art_active_link_names
            or self._art_link_names
            != [str(value) for value in self._art_config["body_names"]]
        ):
            raise ValueError("Articulated manifest and reference topology disagree")
        self._art_observation_variant = env["articulationObservation"]
        if self._art_observation_variant not in {
            "rigid_graph",
            "articulated_graph",
            "joint_state",
            "articulated_graph_joint_state",
        }:
            raise ValueError(
                f"Unknown articulation observation: {self._art_observation_variant}"
            )
        self._art_use_graph = (
            self._art_active_dof_count > 0
            and self._art_observation_variant
            in {"articulated_graph", "articulated_graph_joint_state"}
        )
        self._art_use_joint_state = (
            self._art_active_dof_count > 0
            and self._art_observation_variant
            in {"joint_state", "articulated_graph_joint_state"}
        )
        qvel_scale = np.asarray(
            env["articulationQvelScale"], dtype=np.float32
        ).reshape(-1)
        if qvel_scale.size == 1:
            qvel_scale = np.repeat(qvel_scale, self._art_active_dof_count)
        if qvel_scale.shape != (self._art_active_dof_count,):
            raise ValueError(
                "articulationQvelScale must be scalar or match active object DOFs, got "
                f"{qvel_scale.shape} for {self._art_active_dof_count} active DOFs"
            )
        if not np.all(np.isfinite(qvel_scale)) or np.any(qvel_scale <= 0):
            raise ValueError("articulationQvelScale must contain finite positive values")
        self._art_qvel_scale_np = qvel_scale
        self._art_native_obs_size = int(env["numObs"])
        if self._art_use_joint_state:
            env["numObs"] = self._art_native_obs_size + 4 * self._art_active_dof_count
        if (
            self._art_qref_np.ndim != 2
            or self._art_qref_np.shape[1] != self._art_dof_count
        ):
            raise ValueError(
                "object_joint_qpos must have shape (frames, object DOFs), got "
                f"{self._art_qref_np.shape} for {self._art_dof_count} DOFs"
            )
        if self._art_link_ref_np.shape != (
            self._art_qref_np.shape[0], len(self._art_link_names), 3
        ):
            raise ValueError(
                "object_link_pos must have shape (frames, links, 3), got "
                f"{self._art_link_ref_np.shape}"
            )
        if self._art_link_ref_rot_np.shape != (
            self._art_qref_np.shape[0], len(self._art_link_names), 4
        ):
            raise ValueError(
                "object_link_rot_xyzw must have shape (frames, links, 4), got "
                f"{self._art_link_ref_rot_np.shape}"
            )
        if not np.all(np.isfinite(self._art_qref_np)):
            raise ValueError("Object q reference must be finite")
        if self._art_intended_np.shape != (self._art_qref_np.shape[0], 2):
            raise ValueError("intended contact must have shape (frames, 2)")
        if self._art_contact_points_np.shape != (
            len(self._art_contact_point_links),
            3,
        ):
            raise ValueError("Contact region points and point-link names disagree")
        if (
            not self._art_contact_region_link_names
            or len(set(self._art_contact_region_link_names))
            != len(self._art_contact_region_link_names)
            or set(self._art_contact_point_links)
            != set(self._art_contact_region_link_names)
        ):
            raise ValueError("Canonical contact-region names and points disagree")
        self._art_object_points_np = np.concatenate(
            (parent_points.reshape(-1, 3), child_points.reshape(-1, 3)),
            axis=0,
        )
        self._configure_articulated_actor = configure_articulated_actor
        self._create_static_box_actors = create_static_box_actors
        self._load_articulated_asset = load_articulated_asset
        self._load_static_box_assets = load_static_box_assets
        self._CommonRolloutRecorder = CommonRolloutRecorder
        self._common_human_body_names = SMPLX_BODY_NAMES
        self._art_q_weight = env["articulationRewardWeight"]
        self._art_link_weight = env["articulationLinkRewardWeight"]
        self._art_q_scale = env["articulationRewardScale"]
        self._art_link_scale = env["articulationLinkRewardScale"]
        self._art_rollout_path = env["commonRolloutOutputPath"]
        self._art_rollout_fps = self._art_reference_fps
        if self._art_rollout_fps <= 0.0:
            raise ValueError("articulated reference FPS must be positive")
        if not np.isclose(self._art_rollout_fps, float(env["dataFPS"])):
            raise ValueError("OMOMO dataFPS and articulated reference FPS differ")
        if not np.isclose(
            float(env["plane"]["height"]),
            float(self._art_config["ground_height"]),
        ):
            raise ValueError("author ground plane and articulated manifest differ")
        self._art_humanoid_mjcf_path = Path(
            env["articulatedHumanoidXmlPath"]
        ).expanduser().resolve()
        self._art_humanoid_parents, offsets = _load_humanoid_tree(
            self._art_humanoid_mjcf_path
        )
        self._art_humanoid_offsets_np = offsets
        self._art_recorder = None
        self._art_rollout_next_frame = None
        self._art_rollout_written = False
        self._art_rollout_terminated = None
        self._art_qref = None
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._art_qref = torch.as_tensor(self._art_qref_np, device=self.device)
        self._art_link_local = torch.as_tensor(
            self._art_link_local_np, device=self.device
        )
        self._art_link_local_rot = torch.as_tensor(
            self._art_link_local_rot_np, device=self.device
        )
        self._art_intended = torch.as_tensor(self._art_intended_np, device=self.device)
        self._art_active_dof_ids = torch.as_tensor(
            self._art_active_dof_ids_np,
            device=self.device,
            dtype=torch.long,
        )
        lower = np.asarray(self._target_dof_properties["lower"], dtype=np.float32)
        upper = np.asarray(self._target_dof_properties["upper"], dtype=np.float32)
        active_lower = lower[self._art_active_dof_ids_np]
        active_range = (upper - lower)[self._art_active_dof_ids_np]
        if self._art_active_dof_count and (
            not np.all(np.isfinite(active_range)) or np.any(active_range <= 0)
        ):
            raise ValueError("Active object DOFs require finite positive joint ranges")
        self._art_q_lower = torch.as_tensor(active_lower, device=self.device)
        self._art_q_range = torch.as_tensor(active_range, device=self.device)
        self._art_qvel_scale = torch.as_tensor(
            self._art_qvel_scale_np, device=self.device
        )
        if self._art_rollout_path:
            self._art_rollout_terminated = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
        self._build_articulation_telemetry()
        if self._art_rollout_path:
            if self.num_envs != 1:
                raise ValueError("common rollout recording requires exactly one environment")
            self._art_recorder = self._CommonRolloutRecorder(
                self._art_rollout_path,
                fps=self._art_rollout_fps,
                object_joint_qpos_reference=self._art_qref_np,
                joint_names=self._art_joint_names,
                joint_types=self._art_joint_types,
                intended=self._art_intended_np,
                contact_region_link_names=self._art_target_contact_link_names,
            )

    def _load_target_asset(self):
        asset, properties = self._load_articulated_asset(
            self.gym,
            self.sim,
            self._art_config,
        )
        self._target_asset = [asset]
        self._target_dof_properties = properties
        self._static_box_assets = self._load_static_box_assets(
            self.gym,
            self.sim,
            self._art_config,
        )
        self._target_asset_body_names = list(self.gym.get_asset_rigid_body_names(asset))
        self._studio_agg_bodies = (
            self.gym.get_asset_rigid_body_count(asset) + len(self._static_box_assets)
        )
        self._studio_agg_shapes = (
            self.gym.get_asset_rigid_shape_count(asset) + len(self._static_box_assets)
        )
        self._extra_agg_bodies += self._studio_agg_bodies
        self._extra_agg_shapes += self._studio_agg_shapes
        self.object_points = torch.as_tensor(
            self._art_object_points_np[None],
            device=self.device,
        )

    def _create_ground_plane(self):
        self._add_ground_plane(self.gym, self.sim, self._art_config)

    def _load_play_dataset_body_proxy_assets(self):
        self._extra_agg_bodies -= self._studio_agg_bodies
        self._extra_agg_shapes -= self._studio_agg_shapes
        try:
            super()._load_play_dataset_body_proxy_assets()
        finally:
            self._extra_agg_bodies += self._studio_agg_bodies
            self._extra_agg_shapes += self._studio_agg_shapes

    def _build_target(self, env_id, env_ptr):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self._object_creation_pos_np)
        pose.r = gymapi.Quat(*self._object_creation_rot_np)
        handle = self.gym.create_actor(
            env_ptr,
            self._target_asset[0],
            pose,
            "articulated_object",
            env_id,
            self._object_collision_filter,
            1,
        )
        self._configure_articulated_actor(
            self.gym,
            env_ptr,
            handle,
            self._target_dof_properties,
            self._art_config,
        )
        creation_state = self.gym.get_actor_dof_states(
            env_ptr, handle, gymapi.STATE_ALL
        )
        creation_state = _creation_dof_state(
            creation_state,
            self._art_q0_np,
            self._art_qvel0_np,
        )
        self.gym.set_actor_dof_states(
            env_ptr, handle, creation_state, gymapi.STATE_ALL
        )
        self._target_handles.append(handle)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._validate_humanoid_object_collision_filters(
            self.gym,
            env_ptr,
            self.humanoid_handles[env_id],
        )
        self._create_static_box_actors(
            self.gym,
            env_ptr,
            env_id,
            self._static_box_assets,
            self._art_config,
            collision_filter=self._static_collision_filter,
        )

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        root_view = self._root_states.view(self.num_envs, num_actors, 13)
        self._target_states = root_view[:, 1]
        self._tar_actor_ids = to_torch(
            num_actors * np.arange(self.num_envs) + 1, device=self.device, dtype=torch.int32
        )
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        dof_view = self._dof_state.view(self.num_envs, dofs_per_env, 2)
        self._target_dof_pos = dof_view[:, self.num_dof:self.num_dof + self._art_dof_count, 0]
        self._target_dof_vel = dof_view[:, self.num_dof:self.num_dof + self._art_dof_count, 1]
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        body_view = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        count = len(self._target_asset_body_names)
        self._target_body_state = body_view[:, self.num_bodies:self.num_bodies + count]
        forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self._target_contact_forces = forces.view(self.num_envs, bodies_per_env, 3)[:, self.num_bodies:self.num_bodies + count]
        self._tar_contact_forces = self._target_contact_forces.sum(dim=1)

    def _reset_target(self, env_ids):
        super()._reset_target(env_ids)
        if self._art_qref is None:
            reset_qpos = to_torch(
                self._art_q0_np,
                device=self.device,
            ).expand(env_ids.shape[0], -1)
            reset_qvel = to_torch(
                self._art_qvel0_np,
                device=self.device,
            ).expand(env_ids.shape[0], -1)
        else:
            frames = torch.clamp(
                self.progress_buf[env_ids].long(),
                0,
                self._art_qref.shape[0] - 1,
            )
            reset_qpos = self._art_qref[frames].clone()
            next_frames = torch.clamp(
                frames + 1,
                max=self._art_qref.shape[0] - 1,
            )
            reset_qvel = (
                self._art_qref[next_frames] - self._art_qref[frames]
            ) * self._art_reference_fps
        self._target_dof_pos[env_ids] = reset_qpos
        self._target_dof_vel[env_ids] = reset_qvel
        if self._art_rollout_terminated is not None:
            self._art_rollout_terminated[env_ids] = False

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if self.mode == "test" and self.save_states:
            self._write_reset_body_cache(env_ids)
        if (
            self._art_recorder is None
            or self._art_rollout_written
            or self.num_envs != 1
            or self._art_qref is None
            or self._art_rollout_next_frame is not None
            or not len(env_ids)
            or not torch.any(env_ids == 0)
        ):
            return
        if int(self.progress_buf[0].item()) != 0:
            raise RuntimeError("Formal articulated rollout must reset at reference frame 0")

        # Store frame 0 immediately after the reference reset. The simulator
        # refreshes rigid bodies only after one step, so read the canonical
        # reset body state from hoi_data rather than stale rigid-body tensors.
        data_id = self.data_id[:1]
        frame = self.progress_buf[:1]
        body_state = torch.cat(
            (
                self.extract_data_component("body_pos", ref=True, data_id=data_id, t=frame).reshape(self.num_bodies, 3),
                self.extract_data_component("body_rot", ref=True, data_id=data_id, t=frame).reshape(self.num_bodies, 4),
                self.extract_data_component("body_pos_vel", ref=True, data_id=data_id, t=frame).reshape(self.num_bodies, 3),
                self.extract_data_component("body_rot_vel", ref=True, data_id=data_id, t=frame).reshape(self.num_bodies, 3),
            ),
            dim=1,
        )
        link_count = len(self._art_target_contact_link_names)
        from pipeline.physics.contact import hand_region_distances

        reference_links = _place_links_at_object_root(
            {
                "pos": self._target_states[:1, :3],
                "rot": self._target_states[:1, 3:7],
            },
            {
                "pos": self._art_link_local[frame][
                    :, self._art_contact_point_reference_link_ids
                ],
                "rot": self._art_link_local_rot[frame][
                    :, self._art_contact_point_reference_link_ids
                ],
            },
        )
        points = torch_utils.quat_rotate(
            reference_links["rot"].reshape(-1, 4),
            self._art_contact_points.unsqueeze(0).reshape(-1, 3),
        ).view(1, -1, 3) + reference_links["pos"]
        distance = hand_region_distances(
            body_state[None, self._art_human_contact_ids],
            points,
            self._art_contact_point_link_ids,
            link_count,
            self._art_human_contact_capsule_endpoints,
            self._art_human_contact_capsule_radii,
            self._art_human_contact_capsule_valid,
            self._art_human_contact_box_centers,
            self._art_human_contact_box_quaternions,
            self._art_human_contact_box_half_extents,
            self._art_human_contact_box_valid,
            self._art_human_contact_local_groups,
        )[0]
        self._art_recorder.append(
            0,
            human_root_state=self._humanoid_root_states[0].detach().cpu().numpy(),
            human_dof_pos=self._dof_pos[0].detach().cpu().numpy(),
            human_body_state=body_state.detach().cpu().numpy(),
            object_root_state=self._target_states[0].detach().cpu().numpy(),
            object_joint_qpos=self._target_dof_pos[0].detach().cpu().numpy(),
            region_distance_m=distance.detach().cpu().numpy(),
            hand_force_n=np.zeros(2, dtype=np.float32),
            region_force_n=np.zeros(link_count, dtype=np.float32),
        )
        self._art_rollout_next_frame = 1

    def _write_reset_body_cache(self, env_ids):
        body_count = len(self._art_humanoid_parents)
        if body_count != self.num_bodies or self.num_dof != 3 * (body_count - 1):
            raise ValueError("RePHO humanoid tree and 153-DOF state disagree")
        local_rot = torch.cat(
            (
                self._humanoid_root_states[env_ids, None, 3:7],
                torch_utils.exp_map_to_quat(
                    self._dof_pos[env_ids].reshape(len(env_ids), body_count - 1, 3)
                ),
            ),
            dim=1,
        )
        offsets = torch.as_tensor(
            self._art_humanoid_offsets_np,
            device=self.device,
            dtype=self._rigid_body_pos.dtype,
        )
        positions = [self._humanoid_root_states[env_ids, :3]]
        rotations = [local_rot[:, 0]]
        for body, parent in enumerate(self._art_humanoid_parents[1:], 1):
            parent_rot = rotations[parent]
            positions.append(
                positions[parent]
                + torch_utils.quat_rotate(
                    parent_rot, offsets[body].expand(len(env_ids), -1)
                )
            )
            rotations.append(
                torch_utils.quat_mul(parent_rot, local_rot[:, body])
            )
        pos_end = 3 * body_count
        rot_end = pos_end + 4 * body_count
        self._curr_state_complement[env_ids, 0, :pos_end] = torch.stack(
            positions, dim=1
        ).flatten(1)
        self._curr_state_complement[env_ids, 0, pos_end:rot_end] = torch.stack(
            rotations, dim=1
        ).flatten(1)

    def _reset_env_tensors(self, env_ids):
        human_ids = self._humanoid_actor_ids[env_ids]
        object_ids = self._tar_actor_ids[env_ids]
        actor_ids = torch.stack((human_ids, object_ids), dim=1).reshape(-1).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
        dof_actor_ids = actor_ids if self._art_dof_count else human_ids
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dof_actor_ids),
            len(dof_actor_ids),
        )
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if self._pd_control:
            human = self._action_to_pd_targets(self.actions)
            target = torch.cat(
                (human, torch.zeros_like(self._target_dof_pos)),
                dim=1,
            ).contiguous()
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(target),
                gymtorch.unwrap_tensor(self._humanoid_actor_ids),
                len(self._humanoid_actor_ids),
            )
        else:
            human = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force = torch.cat((human, torch.zeros_like(self._target_dof_pos)), dim=1).contiguous()
            self.gym.set_dof_actuation_force_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(force),
                gymtorch.unwrap_tensor(self._humanoid_actor_ids),
                len(self._humanoid_actor_ids),
            )

    def _reference_frame(self):
        return torch.clamp(self.progress_buf.long(), 0, self._art_qref.shape[0] - 1)

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )

        self._curr_ref_obs[env_ids] = self.hoi_data[
            self.data_id[env_ids], self.progress_buf[env_ids]
        ].clone()
        native = torch.cat(
            (
                self._compute_observations_iter(self.hoi_data, env_ids, 1),
                self._compute_observations_iter(self.hoi_data, env_ids, 16),
            ),
            dim=-1,
        )
        if self._art_use_joint_state:
            native = torch.cat(
                (native, self._art_joint_state_observation(env_ids)), dim=-1
            )
        self.obs_buf[env_ids] = native

    def _compute_observations_iter(self, hoi_data, env_ids=None, delta_t=1):
        if not self._art_use_graph:
            return super()._compute_observations_iter(hoi_data, env_ids, delta_t)
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )

        next_ts = torch.clamp(
            self.progress_buf[env_ids] + delta_t,
            max=self.max_episode_length[self.data_id[env_ids]] - 1,
        )
        ref_obs = hoi_data[self.data_id[env_ids], next_ts].clone()
        obs = torch.cat(
            (
                self._compute_humanoid_obs(env_ids, ref_obs, next_ts),
                self._compute_task_obs(env_ids, ref_obs),
            ),
            dim=-1,
        )
        ig_all, ig, ref_ig = self._art_graph_observation(
            env_ids, ref_obs, next_ts
        )
        return torch.cat((obs, ig_all, ref_ig - ig), dim=-1)

    def _art_joint_state_observation(self, env_ids):
        if self._art_active_dof_count == 0:
            return self._target_dof_pos[env_ids, :0]

        q = self._target_dof_pos[env_ids][:, self._art_active_dof_ids]
        qvel = self._target_dof_vel[env_ids][:, self._art_active_dof_ids]
        frame = self.progress_buf[env_ids]
        frame_1 = torch.clamp(frame + 1, max=self._art_qref.shape[0] - 1)
        frame_16 = torch.clamp(frame + 16, max=self._art_qref.shape[0] - 1)
        normalized_q = 2.0 * (q - self._art_q_lower) / self._art_q_range - 1.0
        return torch.cat(
            (
                normalized_q,
                qvel / self._art_qvel_scale,
                (
                    self._art_qref[frame_1][:, self._art_active_dof_ids] - q
                ) / self._art_q_range,
                (
                    self._art_qref[frame_16][:, self._art_active_dof_ids] - q
                ) / self._art_q_range,
            ),
            dim=-1,
        )

    def _art_graph_observation(self, env_ids, ref_obs, next_ts):
        live_state = self._target_body_state[env_ids][
            :, self._art_contact_point_body_ids
        ]
        local_points = self._art_contact_points.unsqueeze(0).expand(
            len(env_ids), -1, -1
        )
        live_points = torch_utils.quat_rotate(
            live_state[..., 3:7].reshape(-1, 4), local_points.reshape(-1, 3)
        ).view(len(env_ids), -1, 3) + live_state[..., :3]

        ref_links = _place_links_at_object_root(
            {
                "pos": self.extract_data_component("obj_pos", obs=ref_obs),
                "rot": self.extract_data_component("obj_rot", obs=ref_obs),
            },
            {
                "pos": self._art_link_local[next_ts][
                    :, self._art_graph_reference_link_ids
                ],
                "rot": self._art_link_local_rot[next_ts][
                    :, self._art_graph_reference_link_ids
                ],
            },
        )
        ref_points = torch_utils.quat_rotate(
            ref_links["rot"].reshape(-1, 4), local_points.reshape(-1, 3)
        ).view(len(env_ids), -1, 3) + ref_links["pos"]

        live_ig = self._art_encode_graph(
            self._rigid_body_pos[env_ids],
            self._rigid_body_rot[env_ids, 0],
            live_points,
        )
        ref_ig = self._art_encode_graph(
            self.extract_data_component("body_pos", obs=ref_obs).view(
                len(env_ids), -1, 3
            ),
            self.extract_data_component("root_rot", obs=ref_obs),
            ref_points,
        )
        return (
            live_ig.reshape(len(env_ids), -1),
            live_ig[:, self._key_body_ids].reshape(len(env_ids), -1),
            ref_ig[:, self._key_body_ids].reshape(len(env_ids), -1),
        )

    def _art_encode_graph(self, body_pos, root_rot, object_points):
        graph = compute_sdf(body_pos, object_points)
        heading = torch_utils.calc_heading_quat_inv(root_rot)
        heading = heading.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
        graph = torch_utils.quat_rotate(
            heading.reshape(-1, 4), graph.reshape(-1, 3)
        ).view_as(graph)
        norm = torch.linalg.norm(graph, dim=-1, keepdim=True)
        return graph / (norm + 1e-6) * torch.exp(-5.0 * norm)

    def _compute_reset(self):
        super()._compute_reset()
        if not self._art_rollout_path:
            return

        self._art_rollout_terminated |= self._terminate_buf.bool()
        self._terminate_buf[:] = self._art_rollout_terminated.to(self._terminate_buf.dtype)

    def save_run_val(self):
        """Run the author validation command with the Studio articulated task."""

        config = json.loads((Path(self.cam_img_dir) / "config.json").read_text())
        rewards = (self.ref_reward[0].clone() - 7).clamp(min=0).sum(dim=0)
        candidate = rewards.argmax()
        start = 0 if rewards[0] > rewards[candidate] - 10 else candidate.item()
        direction = "backward" if self.reverse_time else "forward"
        if self.reverse_time:
            forward = (
                Path(self.cam_img_dir.replace("backward", "forward"))
                / "ref_tar"
                / f"ref_tar_{self.curr_epoch}"
                / "intermimic.pt"
            )
            while not forward.is_file():
                time.sleep(2)
            time.sleep(10)

        root = Path(config["out_root"]) / f'{config["seq_name"]}_dual' / direction
        command = [
            sys.executable,
            "intermimic/run.py",
            "--task",
            config["studio_task"],
            "--cfg_env",
            config["validation_cfg_env"],
            "--cfg_train",
            config["cfg_train"],
            "--headless",
            "--output_path",
            str(root / "ref_tar" / f"ref_tar_{self.curr_epoch}"),
            "--stateInit",
            "Start",
            "--init_range_left",
            str(start),
            "--reverse_time" if self.reverse_time else "--no_reverse_time",
            "--device_id",
            "0",
            "--rl_device",
            "cuda:0",
            "--motion_file",
            str(Path(config["motion_root"]) / config["seq_name"]),
            "--sub_file_name",
            "intermimic",
            "--checkpoint",
            str(root / "smplx" / "nn" / f"mimic_{self.curr_epoch:08d}.pth"),
            "--hoi_refs_path",
            str(root / "ref_hoi" / f"ref_hoi_{self.curr_epoch}.npz"),
            "--hoi_data_path",
            str(root / "hoi_data" / f"intermimic_{self.curr_epoch}.pt"),
            "--test",
            "--num_envs",
            "1",
            "--save_states",
        ]
        subprocess.run(command, check=True)
        command_path = root / "ref_tar" / f"command_{self.curr_epoch}.txt"
        command_path.parent.mkdir(parents=True, exist_ok=True)
        command_path.write_text(" ".join(command), encoding="utf-8")

    def compute_obj_reward(self, weights):
        native, reset, obj_points, ref_obj_points = super().compute_obj_reward(weights)
        frames = self._reference_frame()
        if self._art_active_dof_count:
            q_reward = torch.exp(
                -self._art_q_scale
                * _mean_normalized_joint_error(
                    {
                        "qpos": self._target_dof_pos[:, self._art_active_dof_ids],
                        "reference": self._art_qref[frames][
                            :, self._art_active_dof_ids
                        ],
                        "range": self._art_q_range,
                    }
                )
            )
        else:
            q_reward = torch.ones(self.num_envs, device=self.device)
        link_reward = torch.ones_like(q_reward)
        if self._art_live_link_ids.numel() > 0:
            live = self._target_body_state[:, self._art_live_link_ids, :3]
            ref_links = _place_links_at_object_root(
                {
                    "pos": self.extract_data_component(
                        "obj_pos", obs=self._curr_ref_obs
                    ),
                    "rot": self.extract_data_component(
                        "obj_rot", obs=self._curr_ref_obs
                    ),
                },
                {
                    "pos": self._art_link_local[frames][
                        :, self._art_ref_link_ids
                    ],
                    "rot": self._art_link_local_rot[frames][
                        :, self._art_ref_link_ids
                    ],
                },
            )
            link_reward = torch.exp(
                -self._art_link_scale
                * torch.mean((live - ref_links["pos"]) ** 2, dim=(1, 2))
            )
        self.extras["articulation_q_reward"] = q_reward
        self.extras["articulation_link_reward"] = link_reward
        reward = native * torch.pow(q_reward, self._art_q_weight) * torch.pow(
            link_reward, self._art_link_weight
        )
        return reward, reset, obj_points, ref_obj_points

    def _build_articulation_telemetry(self):
        target_lookup = {name: i for i, name in enumerate(self._target_asset_body_names)}
        missing_links = [name for name in self._art_link_names if name not in target_lookup]
        if missing_links:
            raise ValueError(f"Object reference links are absent from the loaded URDF: {missing_links}")
        reference_lookup = {
            name: index for index, name in enumerate(self._art_link_names)
        }
        self._art_live_link_ids = torch.as_tensor(
            [target_lookup[name] for name in self._art_active_link_names],
            device=self.device,
            dtype=torch.long,
        )
        self._art_ref_link_ids = torch.as_tensor(
            [reference_lookup[name] for name in self._art_active_link_names],
            device=self.device,
            dtype=torch.long,
        )
        human_names = list(self.gym.get_actor_rigid_body_names(self.envs[0], self.humanoid_handles[0]))
        if tuple(human_names) != self._common_human_body_names:
            raise ValueError("Loaded humanoid bodies do not match the common 52-body order")
        self._art_human_contact_groups = hand2_body_groups(human_names)
        flat_body_ids = tuple(
            body_id for group in self._art_human_contact_groups for body_id in group
        )
        capsule_endpoints, capsule_radii, capsule_valid = load_mjcf_body_capsules(
            self._art_humanoid_mjcf_path,
            [human_names[body_id] for body_id in flat_body_ids],
        )
        box_centers, box_quaternions, box_half_extents, box_valid = load_mjcf_body_boxes(
            self._art_humanoid_mjcf_path,
            [human_names[body_id] for body_id in flat_body_ids],
        )
        if not bool(np.logical_or(capsule_valid, box_valid).all()):
            missing = [
                name
                for name, has_capsule, has_box in zip(
                    [human_names[body_id] for body_id in flat_body_ids],
                    capsule_valid.tolist(),
                    box_valid.tolist(),
                )
                if not has_capsule and not has_box
            ]
            raise ValueError(f"Hand collision bodies have no capsule or box geometry: {missing}")
        self._art_human_contact_ids = torch.as_tensor(
            flat_body_ids, device=self.device, dtype=torch.long
        )
        self._art_human_contact_group_slices = (
            slice(0, len(self._art_human_contact_groups[0])),
            slice(len(self._art_human_contact_groups[0]), len(flat_body_ids)),
        )
        local_body_ids = torch.arange(
            len(flat_body_ids),
            device=self.device,
            dtype=torch.long,
        )
        self._art_human_contact_local_groups = tuple(
            local_body_ids[group]
            for group in self._art_human_contact_group_slices
        )
        self._art_human_contact_capsule_endpoints = torch.as_tensor(
            capsule_endpoints, device=self.device
        )
        self._art_human_contact_capsule_radii = torch.as_tensor(
            capsule_radii, device=self.device
        )
        self._art_human_contact_capsule_valid = torch.as_tensor(
            capsule_valid, device=self.device
        )
        self._art_human_contact_box_centers = torch.as_tensor(
            box_centers, device=self.device
        )
        self._art_human_contact_box_quaternions = torch.as_tensor(
            box_quaternions, device=self.device
        )
        self._art_human_contact_box_half_extents = torch.as_tensor(
            box_half_extents, device=self.device
        )
        self._art_human_contact_box_valid = torch.as_tensor(
            box_valid, device=self.device
        )
        region_names = tuple(self._art_contact_region_link_names)
        missing_region_links = sorted(set(region_names).difference(target_lookup))
        if missing_region_links:
            raise ValueError(f"Contact region links are absent from the loaded URDF: {missing_region_links}")
        self._art_target_contact_link_names = region_names
        target_local = [target_lookup[name] for name in region_names]
        self._art_target_contact_body_ids = torch.as_tensor(
            target_local, device=self.device, dtype=torch.long
        )
        missing_point_links = [
            name for name in self._art_contact_point_links if name not in target_lookup
        ]
        if missing_point_links:
            raise ValueError("A contact-region point refers to an unresolved object link")
        point_body_ids = [target_lookup[name] for name in self._art_contact_point_links]
        self._art_contact_points = torch.as_tensor(self._art_contact_points_np, device=self.device)
        self._art_contact_point_body_ids = torch.as_tensor(
            point_body_ids, device=self.device, dtype=torch.long
        )
        point_link_lookup = {
            name: index
            for index, name in enumerate(self._art_target_contact_link_names)
        }
        self._art_contact_point_link_ids = torch.as_tensor(
            [point_link_lookup[name] for name in self._art_contact_point_links],
            device=self.device,
            dtype=torch.long,
        )
        missing_reference_links = sorted(
            set(self._art_contact_point_links).difference(reference_lookup)
        )
        if missing_reference_links:
            raise ValueError(
                "Contact-region links are absent from object reference: "
                f"{missing_reference_links}"
            )
        self._art_contact_point_reference_link_ids = torch.as_tensor(
            [reference_lookup[name] for name in self._art_contact_point_links],
            device=self.device,
            dtype=torch.long,
        )
        if self._art_use_graph:
            if not self._art_contact_point_links:
                raise ValueError("articulated_graph requires contact-region points")
            self._art_graph_reference_link_ids = torch.as_tensor(
                [reference_lookup[name] for name in self._art_contact_point_links],
                device=self.device,
                dtype=torch.long,
            )

    def _contact_telemetry(self):
        link_count = len(self._art_target_contact_link_names)
        distance = torch.full(
            (self.num_envs, 2, link_count), float("inf"), device=self.device
        )
        if self._art_contact_points.numel() > 0:
            state = self._target_body_state[:, self._art_contact_point_body_ids]
            local = self._art_contact_points.unsqueeze(0).expand(self.num_envs, -1, -1)
            points = torch_utils.quat_rotate(state[..., 3:7].reshape(-1, 4), local.reshape(-1, 3))
            points = points.view(self.num_envs, -1, 3) + state[..., :3]
            human_state = self._rigid_body_state.view(self.num_envs, -1, 13)[
                :, self._art_human_contact_ids
            ]
            for link_index in range(link_count):
                link_points = points[
                    :, self._art_contact_point_link_ids == link_index
                ]
                capsule_distance = capsule_region_surface_distances(
                    body_pos=human_state[..., :3],
                    body_quat_xyzw=human_state[..., 3:7],
                    endpoints_local=self._art_human_contact_capsule_endpoints,
                    radii=self._art_human_contact_capsule_radii,
                    valid=self._art_human_contact_capsule_valid,
                    region_points=link_points,
                )
                box_distance = box_region_surface_distances(
                    body_pos=human_state[..., :3],
                    body_quat_xyzw=human_state[..., 3:7],
                    centers_local=self._art_human_contact_box_centers,
                    quaternions_local_xyzw=self._art_human_contact_box_quaternions,
                    half_extents=self._art_human_contact_box_half_extents,
                    valid=self._art_human_contact_box_valid,
                    region_points=link_points,
                )
                body_distance = torch.minimum(capsule_distance, box_distance)
                for label, group in enumerate(self._art_human_contact_group_slices):
                    distance[:, label, link_index] = body_distance[:, group].min(dim=1).values
        hand_force = torch.stack(
            tuple(
                torch.linalg.norm(self._contact_forces[:, group], dim=-1).amax(dim=1)
                for group in self._art_human_contact_groups
            ),
            dim=1,
        )
        region_force = torch.linalg.norm(
            self._target_contact_forces[:, self._art_target_contact_body_ids], dim=-1
        )
        return distance, hand_force, region_force

    def post_physics_step(self):
        try:
            super().post_physics_step()
        except SystemExit:
            self._record_common_rollout_step()
            raise
        self._record_common_rollout_step()

    def _record_common_rollout_step(self):
        if self._art_recorder is None or self._art_rollout_written:
            return
        distance, hand_force, region_force = self._contact_telemetry()
        frame = int(self._reference_frame()[0].item())
        if self._art_rollout_next_frame is None:
            return
        if frame != self._art_rollout_next_frame:
            raise RuntimeError(
                "RePHO recorder expected reference frame "
                f"{self._art_rollout_next_frame}, got {frame}"
            )
        body_state = self._rigid_body_state.view(self.num_envs, -1, 13)[0, :self.num_bodies]
        self._art_recorder.append(
            frame,
            human_root_state=self._humanoid_root_states[0].detach().cpu().numpy(),
            human_dof_pos=self._dof_pos[0].detach().cpu().numpy(),
            human_body_state=body_state.detach().cpu().numpy(),
            object_root_state=self._target_states[0].detach().cpu().numpy(),
            object_joint_qpos=self._target_dof_pos[0].detach().cpu().numpy(),
            region_distance_m=distance[0].detach().cpu().numpy(),
            hand_force_n=hand_force[0].detach().cpu().numpy(),
            region_force_n=region_force[0].detach().cpu().numpy(),
        )
        self._art_rollout_next_frame += 1
        if bool(self.reset_buf[0].item()) or frame >= self._art_qref.shape[0] - 1:
            self._art_recorder.seal()
            self._art_rollout_written = True


__all__ = ["RePHOArticulated"]
