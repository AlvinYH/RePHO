#!/bin/bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
seq_name=$1
gpu_id=$2
out_root=$3
motion_root=$4
cfg_env=$5
cfg_train=$6
current_epoch=$7
init_range_left=$8

CUDA_VISIBLE_DEVICES=${gpu_id} python intermimic/run.py \
    --task InterMimic \
    --cfg_env ${cfg_env} \
    --cfg_train ${cfg_train} \
    --headless \
    --output_path ${out_root}/${seq_name}_dual/backward/ref_tar/ref_tar_${current_epoch} \
    --stateInit Start \
    --init_range_left ${init_range_left} \
    --reverse_time \
    --device_id 0 \
    --rl_device cuda:0 \
    --motion_file ${motion_root}/${seq_name} \
    --sub_file_name intermimic_vistracker \
    --checkpoint ${out_root}/${seq_name}_dual/backward/smplx/nn/mimic_000${current_epoch}.pth \
    --hoi_refs_path ${out_root}/${seq_name}_dual/backward/ref_hoi/ref_hoi_${current_epoch}.npz \
    --hoi_data_path ${out_root}/${seq_name}_dual/backward/hoi_data/intermimic_${current_epoch}.pt \
    --test --num_envs 1 --save_states
    
