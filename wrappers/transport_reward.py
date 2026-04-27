import numpy as np


def transport_dense_reward(env, obs_dict):
    """Staged shaped reward for TwoArmTransport.

    Stages (robot0 handles payload, robot1 handles trash, in parallel):
      - robot0 reaches payload in start bin
      - payload transits to target bin
      - robot1 reaches trash in target bin
      - trash transits to trash bin
    Each distance term uses 1 - tanh(scale * d) to saturate near contact.
    Phase-completion indicators (payload_in_target_bin, trash_in_trash_bin)
    give step bonuses so GAE can credit them. A final bonus fires on success.
    """
    transport = env.transport

    eef0 = np.asarray(obs_dict["robot0_eef_pos"])
    eef1 = np.asarray(obs_dict["robot1_eef_pos"])
    payload = np.asarray(transport.payload_pos)
    trash = np.asarray(transport.trash_pos)
    target_bin = np.asarray(transport.target_bin_pos)
    trash_bin = np.asarray(transport.trash_bin_pos)

    payload_done = float(transport.payload_in_target_bin)
    trash_done = float(transport.trash_in_trash_bin)

    d0_payload = np.linalg.norm(eef0 - payload)
    d_payload_target = np.linalg.norm(payload - target_bin)
    d1_trash = np.linalg.norm(eef1 - trash)
    d_trash_bin = np.linalg.norm(trash - trash_bin)

    r_reach_payload = 1.0 - np.tanh(3.0 * d0_payload)
    r_place_payload = 1.0 - np.tanh(3.0 * d_payload_target)
    r_reach_trash = 1.0 - np.tanh(3.0 * d1_trash)
    r_place_trash = 1.0 - np.tanh(3.0 * d_trash_bin)

    reward = (
        0.10 * r_reach_payload
        + 0.30 * r_place_payload
        + 0.10 * r_reach_trash
        + 0.30 * r_place_trash
        + 0.50 * payload_done
        + 0.50 * trash_done
    )
    if env._check_success():
        reward += 5.0
    return float(reward)
