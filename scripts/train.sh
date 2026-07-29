#!/bin/bash
set -e

seq_name=$1
forward_gpu_id=$2
backward_gpu_id=${3:-${forward_gpu_id}}
out_root=${4:-tracking_intermediate_file}
motion_root=${5:-data/demo_data/output}
cfg_env=${6:-intermimic/data/cfg/omomo_train_917.yaml}
cfg_train=${7:-intermimic/data/cfg/train/rlg/omomo.yaml}
task=${8:-InterMimic}
sub_file_name=${9:-intermimic_vistracker}
validation_cfg_env=${10:-${cfg_env}}
forward_cfg_env=${11:-${cfg_env}}
backward_cfg_env=${12:-${cfg_env}}
forward_cfg_train=${13:-${cfg_train}}
backward_cfg_train=${14:-${cfg_train}}

bash scripts/train_dual_forward.sh \
    "${seq_name}" "${forward_gpu_id}" "${out_root}" "${motion_root}" \
    "${forward_cfg_env}" "${forward_cfg_train}" "${task}" "${sub_file_name}" \
    "${validation_cfg_env}" &
forward_pid=$!
sleep 30
bash scripts/train_dual_backward.sh \
    "${seq_name}" "${backward_gpu_id}" "${out_root}" "${motion_root}" \
    "${backward_cfg_env}" "${backward_cfg_train}" "${task}" "${sub_file_name}" \
    "${validation_cfg_env}"
wait "${forward_pid}"
