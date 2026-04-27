#!/bin/bash
/home/abhi/miniconda3/envs/dsrl/bin/python eval_mpc.py \
  --bc_checkpoint /home/abhi/dsrl/checkpoints/bc/transport/state_3000.pt \
  --wm_checkpoint checkpoints/wm/transport/wm.pt \
  --n_samples 8 \
  --mpc_horizon 1 \
  --n_episodes 20
