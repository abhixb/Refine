import json
import multiprocessing as mp
import h5py
import numpy as np
import robosuite as suite


def _hdf5_to_env_key(k):
    return "object-state" if k == "object" else k


def _extract_obs(obs_dict, obs_keys):
    env_keys = [_hdf5_to_env_key(k) for k in obs_keys]
    parts = [np.array(obs_dict[k], dtype=np.float32).flatten() for k in env_keys]
    return np.concatenate(parts)


def _make_env(hdf5_path, horizon):
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    env_kwargs = env_args["env_kwargs"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["reward_shaping"] = False
    env_kwargs["horizon"] = horizon
    env_kwargs.pop("ignore_done", None)
    return suite.make(env_args["env_name"], **env_kwargs)


def _resolve_reward(task):
    from wrappers.transport_reward import transport_dense_reward
    return {"transport": transport_dense_reward}[task]


def _worker(pipe, hdf5_path, horizon, obs_keys, task, pred_horizon, action_per_step):
    env = _make_env(hdf5_path, horizon)
    reward_fn = _resolve_reward(task)
    step_count = 0

    while True:
        cmd, data = pipe.recv()
        if cmd == "reset":
            obs_dict = env.reset()
            step_count = 0
            pipe.send(_extract_obs(obs_dict, obs_keys))

        elif cmd == "step":
            action_chunk = data.reshape(pred_horizon, action_per_step)
            chunk_reward = 0.0
            chunk_done = False
            chunk_success = False
            obs_dict = None

            for a in action_chunk:
                a = np.clip(a, -1, 1)
                obs_dict, _, env_done, _ = env.step(a)
                chunk_reward += reward_fn(env, obs_dict)
                step_count += 1
                if env._check_success():
                    chunk_success = True
                    chunk_done = True
                    break
                if env_done or step_count >= horizon:
                    chunk_done = True
                    break

            pipe.send((_extract_obs(obs_dict, obs_keys), chunk_reward, chunk_done, chunk_success))

        elif cmd == "close":
            env.close()
            pipe.close()
            return


class ParallelEnvs:
    def __init__(self, n_envs, hdf5_path, horizon, obs_keys, task,
                 pred_horizon, action_per_step):
        self.n = n_envs
        ctx = mp.get_context("spawn")
        self.pipes = []
        self.processes = []
        for _ in range(n_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(child, hdf5_path, horizon, obs_keys, task,
                      pred_horizon, action_per_step),
                daemon=True,
            )
            p.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(p)

    def reset_all(self):
        for pipe in self.pipes:
            pipe.send(("reset", None))
        return [pipe.recv() for pipe in self.pipes]

    def step_send(self, idx, action_flat):
        self.pipes[idx].send(("step", action_flat))

    def step_recv(self, idx):
        return self.pipes[idx].recv()

    def reset_one(self, idx):
        self.pipes[idx].send(("reset", None))
        return self.pipes[idx].recv()

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
