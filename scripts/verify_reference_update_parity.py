#!/usr/bin/env python3
"""GPU parity check for the tensorized RePHO reference update."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

from isaacgym import gymapi
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "intermimic"))

from env.tasks.intermimic import (
    InterMimic,
    _atomic_savez,
    _atomic_torch_save,
    compute_sdf,
)


def _old_update(task) -> None:
    reset_ind = task.reset_buf == 1
    data_id = task.data_id[reset_ind]
    max_episode_length = task.max_episode_length[data_id]
    start_index = task.start_times[reset_ind]
    end_index = task.progress_buf[reset_ind]
    task.to_end_cnt += (
        (end_index >= max_episode_length[0] - 1)
        & (start_index < max_episode_length[0] - 50)
    ).sum().item()
    task.middle_to_end_cnt += (
        (end_index >= max_episode_length[0] - 1)
        & (start_index < max_episode_length[0] // 2)
    ).sum().item()
    task.left_to_end_cnt += (
        (end_index >= max_episode_length[0] - 1)
        & (start_index <= task._init_range_left)
    ).sum().item()
    if task.left_to_end_cnt > 200:
        task._init_range_left = 0
    curr_reward = task._curr_reward[reset_ind]
    task._sum_reward[reset_ind] = 0
    task._curr_reward[reset_ind] = 0
    state = task._curr_state[reset_ind]
    reward = torch.zeros(
        (curr_reward.shape[0], task.hoi_refs.shape[0], task.hoi_refs.shape[2]),
        device=curr_reward.device,
    )
    reward_opposite = torch.zeros_like(reward)
    for row in range(curr_reward.shape[0]):
        contact_obj = task.extract_data_component(
            "contact_obj",
            obs=task.hoi_data[0, start_index[row]:end_index[row]],
        )
        contact_obj_whole = task.extract_data_component(
            "contact_obj",
            obs=task.hoi_data[0, 0:task.max_episode_length[0]],
        )
        contact_obj_expand = task.extract_data_component(
            "contact_obj",
            obs=task.hoi_data[
                0,
                max(0, start_index[row] - 20):
                min(
                    task.max_episode_length[0] - 2,
                    end_index[row] + 20,
                ),
            ],
        )
        long_enough_1 = (
            end_index[row] - start_index[row] > 30
        ) and (
            torch.all(contact_obj_expand > 0.1)
            or torch.all(contact_obj_whole < 0.1)
        )
        long_enough_2 = (
            end_index[row] - start_index[row] > 70
        ) and (
            torch.sum(contact_obj) > 70
            or torch.all(contact_obj_whole < 0.1)
        )
        if long_enough_1 or long_enough_2:
            if (
                task.to_end_cnt > 50
                and end_index[row] >= max_episode_length[0] - 1
            ):
                values = torch.arange(
                    0,
                    end_index[row] - start_index[row] + 1,
                    device=start_index.device,
                ).flip(0)
                reward[
                    row,
                    data_id[row],
                    start_index[row]:end_index[row] + 1,
                ] = values
                if end_index[row] - start_index[row] > 60:
                    reward_opposite[
                        row,
                        data_id[row],
                        start_index[row]:end_index[row] + 1,
                    ] = values.flip(0)
            else:
                values = torch.arange(
                    20,
                    end_index[row] - start_index[row] + 1,
                    device=start_index.device,
                ).flip(0)
                reward[
                    row,
                    data_id[row],
                    start_index[row]:end_index[row] - 20 + 1,
                ] = values
                if end_index[row] - start_index[row] > 60:
                    reward_opposite[
                        row,
                        data_id[row],
                        start_index[row]:end_index[row] - 20 + 1,
                    ] = values.flip(0) - 10
        elif torch.all(contact_obj_expand > 0.1):
            reward[row, data_id[row], start_index[row]] = (
                end_index[row] - start_index[row]
            )
    adjust_reward, winner = reward.max(dim=0)
    adjust_opposite, winner_opposite = reward_opposite.max(dim=0)
    ratio = max((53250 - task._stage_epoch()) / 1000, 0)
    current_contact = task._curr_contact[reset_ind]
    for motion in range(reward.shape[1]):
        for frame in range(reward.shape[2]):
            value, slot = task.ref_reward[motion, 1:, frame].min(dim=0)
            slot = slot + 1
            source = winner[motion, frame]
            source_frame = frame - start_index[source]
            if source_frame >= 0:
                stop = end_index[source] - frame + 1
                sum_reward = (
                    curr_reward[source, :stop] * task.powers[:stop]
                ).sum()
            else:
                sum_reward = 0
            if source_frame >= 0 and (
                (
                    adjust_reward[motion, frame] > value
                    and sum_reward
                    >= task.ref_reward_sum[motion, slot, frame] * ratio
                )
                or adjust_reward[motion, frame] > value + 10
            ):
                task.ref_reward[motion, slot, frame] = adjust_reward[
                    motion, frame
                ]
                task.ref_reward_sum[motion, slot, frame] = sum_reward
                task.hoi_refs[motion, slot, frame] = state[
                    source, source_frame
                ]
                if source_frame > 0:
                    task.contact_refs[motion, slot, frame] = current_contact[
                        source, source_frame
                    ]
            value, slot = task.ref_reward_for_opposite[
                motion, 1:, frame
            ].min(dim=0)
            slot = slot + 1
            source = winner_opposite[motion, frame]
            source_frame = frame - start_index[source]
            if source_frame >= 0 and adjust_opposite[motion, frame] > value:
                task.ref_reward_for_opposite[
                    motion, slot, frame
                ] = adjust_opposite[motion, frame]
                task.hoi_refs_for_opposite[
                    motion, slot, frame
                ] = state[source, source_frame]
    task.ref_reward[:, 1:, :] = (
        task.ref_reward[:, 1:, :] * (1 - 5e-4)
    )
    task.ref_reward_for_opposite[:, 1:, :] = (
        task.ref_reward_for_opposite[:, 1:, :] * (1 - 5e-4)
    )
    task.ref_reward_sum = task.ref_reward_sum * (1 - 5e-2)


def _task(seed: int, contact_mode: str):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    envs, frames, slots, state_features, contact_features = 8, 128, 5, 9, 4
    starts = torch.tensor(
        [0, 0, 5, 10, 20, 1, 15, 37],
        device=device,
    )
    durations = torch.tensor(
        [30, 31, 60, 61, 70, 71, 74, 90],
        device=device,
    )
    ends = starts + durations
    if contact_mode == "none":
        contact = torch.zeros((frames, 1), device=device)
    elif contact_mode == "all":
        contact = torch.ones((frames, 1), device=device)
    else:
        contact = torch.zeros((frames, 1), device=device)
        contact[8:105] = 1
    reset = torch.ones(envs, dtype=torch.long, device=device)
    task = SimpleNamespace(
        mode="train",
        reset_buf=reset,
        _terminate_buf=torch.zeros_like(reset),
        progress_buf=ends,
        obs_buf=torch.zeros((envs, 1), device=device),
        _rigid_body_pos=torch.zeros((envs, 1, 3), device=device),
        max_episode_length=torch.tensor([frames], device=device),
        data_id=torch.zeros(envs, dtype=torch.long, device=device),
        _enable_early_termination=True,
        _termination_heights=torch.zeros(1, device=device),
        start_times=starts,
        rollout_length=frames,
        kinematic_reset=torch.zeros(envs, dtype=torch.bool, device=device),
        contact_reset=torch.zeros((envs, 5), device=device),
        psi=5,
        to_end_cnt=49,
        middle_to_end_cnt=0,
        left_to_end_cnt=0,
        _init_range_left=0,
        _sum_reward=torch.rand(
            envs, generator=generator, device=device
        ),
        _curr_reward=torch.rand(
            (envs, frames), generator=generator, device=device
        ),
        _curr_state=torch.rand(
            (envs, frames, state_features),
            generator=generator,
            device=device,
        ),
        _curr_contact=torch.rand(
            (envs, frames, contact_features),
            generator=generator,
            device=device,
        ),
        hoi_data=contact.unsqueeze(0),
        hoi_refs=torch.rand(
            (1, slots, frames, state_features),
            generator=generator,
            device=device,
        ),
        contact_refs=torch.rand(
            (1, slots, frames, contact_features),
            generator=generator,
            device=device,
        ),
        ref_reward=torch.rand(
            (1, slots, frames), generator=generator, device=device
        ) * 15,
        ref_reward_sum=torch.rand(
            (1, slots, frames), generator=generator, device=device
        ) * 20,
        hoi_refs_for_opposite=torch.rand(
            (1, slots, frames, state_features),
            generator=generator,
            device=device,
        ),
        contact_refs_for_opposite=torch.rand(
            (1, slots, frames, contact_features),
            generator=generator,
            device=device,
        ),
        ref_reward_for_opposite=torch.rand(
            (1, slots, frames), generator=generator, device=device
        ) * 15,
        powers=0.99 ** torch.arange(frames, device=device),
        just_update_tar=False,
        _state_init=InterMimic.StateInit.Hybrid,
    )
    task.compute_hoi_reset = lambda *args: (
        reset.clone(),
        torch.zeros_like(reset),
    )
    task.extract_data_component = lambda name, obs: obs
    task._stage_epoch = lambda epoch=None: 53000
    return task


def _clone(task):
    values = {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in vars(task).items()
    }
    clone = SimpleNamespace(**values)
    reset = clone.reset_buf
    clone.compute_hoi_reset = lambda *args: (
        reset.clone(),
        torch.zeros_like(reset),
    )
    clone.extract_data_component = lambda name, obs: obs
    clone._stage_epoch = lambda epoch=None: 53000
    return clone


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tensor_names = (
        "_sum_reward",
        "_curr_reward",
        "ref_reward",
        "ref_reward_sum",
        "hoi_refs",
        "contact_refs",
        "ref_reward_for_opposite",
        "hoi_refs_for_opposite",
    )
    scalar_names = (
        "to_end_cnt",
        "middle_to_end_cnt",
        "left_to_end_cnt",
        "_init_range_left",
    )
    for contact_mode in ("none", "all", "mixed"):
        for seed in range(8):
            source = _task(seed, contact_mode)
            old = _clone(source)
            new = _clone(source)
            _old_update(old)
            InterMimic._compute_reset(new)
            for name in tensor_names:
                if not torch.equal(getattr(old, name), getattr(new, name)):
                    difference = (
                        getattr(old, name) - getattr(new, name)
                    ).abs().max().item()
                    raise AssertionError(
                        f"{name} mismatch at {contact_mode}/{seed}: {difference}"
                    )
            for name in scalar_names:
                if getattr(old, name) != getattr(new, name):
                    raise AssertionError(
                        f"{name} mismatch at {contact_mode}/{seed}"
                    )
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        arrays = {
            "ref_reward": torch.randn(
                (2, 5, 17), device="cuda"
            ).cpu().numpy(),
            "hoi_refs": torch.randn(
                (2, 5, 17, 9), device="cuda"
            ).cpu().numpy(),
        }
        old_npz = root / "old.npz"
        new_npz = root / "new.npz"
        np.savez(old_npz, **arrays)
        _atomic_savez(str(new_npz), **arrays)
        with np.load(old_npz) as old_values, np.load(new_npz) as new_values:
            for name in arrays:
                if not np.array_equal(old_values[name], new_values[name]):
                    raise AssertionError(f"atomic NPZ mismatch: {name}")
        tensor = torch.randn((17, 9), device="cuda").cpu()
        old_pt = root / "old.pt"
        new_pt = root / "new.pt"
        torch.save(tensor, old_pt)
        _atomic_torch_save(tensor, str(new_pt))
        if not torch.equal(
            torch.load(old_pt, weights_only=False),
            torch.load(new_pt, weights_only=False),
        ):
            raise AssertionError("atomic torch artifact mismatch")
        if list(root.glob("*.tmp")):
            raise AssertionError("atomic publisher left temporary files")
    points1 = torch.randn((5, 17, 3), device="cuda")
    points2 = torch.randn((5, 11, 3), device="cuda")
    points2[:, 1] = points2[:, 0]
    distances = points1.unsqueeze(2) - points2.unsqueeze(1)
    nearest = torch.argmin(torch.norm(distances, dim=-1), dim=-1)
    batch, point = torch.meshgrid(
        torch.arange(points1.shape[0]),
        torch.arange(points1.shape[1]),
        indexing="ij",
    )
    expected_sdf = distances[batch, point, nearest].contiguous()
    if not torch.equal(expected_sdf, compute_sdf(points1, points2)):
        raise AssertionError("compute_sdf gather mismatch")
    old_curr = torch.randn((32, 97), device="cuda")
    old_hist = torch.randn_like(old_curr)
    new_curr = old_curr.clone()
    new_hist = old_hist.clone()
    for step in range(16):
        if step % 3 == 0:
            rows = torch.arange(step % 7, 32, 7, device="cuda")
            old_hist[rows] = 0
            new_hist[rows] = 0
        old_hist = old_curr.clone()
        new_hist, new_curr = new_curr, new_hist
        next_obs = torch.randn_like(old_curr)
        old_curr[:] = next_obs
        new_curr[:] = next_obs
        if not torch.equal(old_hist, new_hist):
            raise AssertionError(f"history buffer mismatch at step {step}")
        if not torch.equal(old_curr, new_curr):
            raise AssertionError(f"current buffer mismatch at step {step}")
    print("RePHO reference update: exact CUDA tensor parity")
    print("RePHO atomic artifacts: exact tensor/array parity")
    print("RePHO SDF/history buffers: exact CUDA tensor parity")


if __name__ == "__main__":
    main()
