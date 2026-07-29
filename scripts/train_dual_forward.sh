#!/bin/bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
seq_name=$1
gpu_id=$2
out_root=${3:-tracking_intermediate_file}
motion_root=${4:-data/demo_data/output}
cfg_env=${5:-intermimic/data/cfg/omomo_train_917.yaml}
cfg_train=${6:-intermimic/data/cfg/train/rlg/omomo.yaml}
task=${7:-InterMimic}
sub_file_name=${8:-intermimic_vistracker}
validation_cfg_env=${9:-${cfg_env}}

config_json="${out_root}/${seq_name}_dual/forward/config.json"
mkdir -p "${out_root}/${seq_name}_dual/forward"

if [ "${task}" = "InterMimic" ]; then
cat > "${config_json}" << EOF
{
  "seq_name": "${seq_name}",
  "gpu_id": "${gpu_id}",
  "out_root": "${out_root}",
  "motion_root": "${motion_root}",
  "cfg_env": "${cfg_env}",
  "cfg_train": "${cfg_train}"
}
EOF
else
cat > "${config_json}" << EOF
{
  "seq_name": "${seq_name}",
  "gpu_id": "${gpu_id}",
  "out_root": "${out_root}",
  "motion_root": "${motion_root}",
  "cfg_env": "${cfg_env}",
  "cfg_train": "${cfg_train}",
  "validation_cfg_env": "${validation_cfg_env}",
  "studio_task": "${task}"
}
EOF
fi

CUDA_VISIBLE_DEVICES=${gpu_id} python intermimic/run.py \
    --task "${task}" \
    --cfg_env ${cfg_env} \
    --cfg_train ${cfg_train} \
    --headless \
    --output_path ${out_root}/${seq_name}_dual/forward \
    --stateInit Traverse_Random \
    --no_reverse_time \
    --device_id 0 \
    --rl_device cuda:0 \
    --motion_file ${motion_root}/${seq_name} \
    --sub_file_name "${sub_file_name}"
