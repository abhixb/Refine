# Refine

Diffusion behavioral cloning + DSRL-style RL fine-tuning on Robomimic.

A diffusion policy is first trained with BC on demonstrations, then refined by SAC operating in the diffusion policy's noise space (Diffusion Steering RL). The frozen diffusion model maps SAC's noise vectors `w` to environment actions via DDIM, so RL learns to *steer* an already-good prior instead of learning control from scratch.

## Results (Robomimic, low-dim, proficient-human)

Diffusion BC alone:

| Task   | Success rate |
|--------|--------------|
| Lift   | 98%          |
| Can    | 96%          |
| Square | 82%          |

Evaluated over 20 episodes per task with DDIM (10 steps), `pred_horizon=4`.

## Layout

```
config/         per-task BC + DSRL hyperparameters
data/           Robomimic hdf5 loader and min-max normalizer
model/          diffusion policy (MLP epsilon predictor, EMA, DDIM sampler)
wrappers/       DSRLEnvWrapper - exposes noise space to SB3 SAC
train_bc.py     stage 1: BC pretraining of the diffusion policy
train_dsrl.py   stage 2: SAC in noise space over the frozen diffusion policy
evaluate.py     rollout the BC diffusion policy
record_eval.py  same, with video recording
```

## Stage 1 - BC pretraining

```
python train_bc.py \
    --data_path datasets/lift/ph/low_dim_v141.hdf5 \
    --save_dir checkpoints/bc/lift \
    --task lift \
    --pred_horizon 4 \
    --diffusion_steps 100 \
    --hidden_dims 1024 1024 1024 \
    --epochs 3000
```

Per-task configs live in `config/bc_{lift,can,square}.yaml`. Checkpoints store the model, optimizer, EMA, both normalizers, observation keys, and the full arg dict so downstream stages can reconstruct everything.

Evaluate:

```
python evaluate.py --checkpoint checkpoints/bc/lift/state_3000.pt --n_episodes 20 --ddim_steps 10
```

## Stage 2 - DSRL fine-tuning

SAC's action space becomes the diffusion noise vector `w` of shape `pred_horizon * action_per_step`. Each wrapper step denoises `w` into a chunk of `pred_horizon` env actions and executes them open-loop. The replay buffer is warm-started by rolling out the BC policy with `w ~ N(0, I)`.

```
python train_dsrl.py \
    --bc_checkpoint checkpoints/bc/lift/state_3000.pt \
    --task lift \
    --action_magnitude 1.0 \
    --ddim_steps 10 \
    --initial_steps 320000 \
    --train_steps 100000 \
    --utd 20
```

Best checkpoint by online success rate is written to `checkpoints/dsrl/<task>/sac_best.zip`.

## Requirements

- PyTorch (CUDA recommended)
- robosuite, robomimic
- stable-baselines3, gymnasium
- h5py, numpy

Datasets: standard Robomimic `low_dim_v141.hdf5` files under `datasets/<task>/ph/`.
