import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import supervised_gym as sg

def test_runner():
    hyps = {
        "exp_len": 20,
        "n_envs": 3,
        "n_eval_steps": 16,
        "batch_size": 16,
        "n_frame_stack": 1,
        "seq_len": 7,
        "randomize_order": False,
        "env_type": "gordongames-v0",
        "oracle_type": "GordonOracle",
        "seed": 0,
        "runner_seed_offset": 1,
        "preprocessor": "null_prep",
        "harsh": True,
        "pixel_density": 1,
        "grid_size": (9,9),
        "targ_range": [0,5],
    }
    hyps["obs_shape"] = tuple(
        [hyps["n_frame_stack"],
        *hyps["grid_size"]]
    )
    exp = sg.experience.ExperienceReplay(**hyps)
    start_q = mp.Queue(1)
    stop_q = mp.Queue(1)
    env = sg.envs.SequentialEnvironment(**hyps)
    runner = sg.experience.Runner(
        exp.shared_exp,
        hyps,
        start_q,
        stop_q
    )
    proc = mp.Process(target=runner.run)
    proc.start()
    start_q.put(0)
    stop_q.get()
    for i in range(exp.shared_exp["obs"].shape[1]):
        print("done:", exp.shared_exp["dones"][0][i].item())
        print("rew:", exp.shared_exp["rews"][0][i].item())
        print("actn:", exp.shared_exp["actns"][0][i].item())
        print()
        plt.imshow(exp.shared_exp["obs"][0][i].numpy().squeeze())
        plt.show()
    proc.terminate()

def test_validation_runner():
    hyps = {
        "exp_len": 20,
        "n_envs": 3,
        "n_eval_steps": 16,
        "batch_size": 16,
        "n_frame_stack": 1,
        "seq_len": 7,
        "randomize_order": False,
        "env_type": "gordongames-v0",
        "oracle_type": "GordonOracle",
        "seed": 0,
        "runner_seed_offset": 1,
        "preprocessor": "null_prep",
        "harsh": True,
        "pixel_density": 1,
        "grid_size": (9,9),
        "targ_range": [0,5],
        "render": True,
    }
    hyps["obs_shape"] = tuple(
        [hyps["n_frame_stack"],
        *hyps["grid_size"]]
    )
    val_runner = sg.experience.ValidationRunner(hyps)
    model = sg.models.RandomModel(actn_size=val_runner.env.actn_size)
    eval_data = val_runner.rollout(
        model,
        hyps["n_eval_steps"]
    )

def test_data_collector():
    hyps = {
        "exp_len": 40,
        "n_envs": 3,
        "n_eval_steps": 16,
        "batch_size": 16,
        "n_frame_stack": 1,
        "seq_len": 7,
        "randomize_order": False,
        "env_type": "gordongames-v0",
        "oracle_type": "GordonOracle",
        "seed": 0,
        "preprocessor": "null_prep",
        "harsh": True,
        "pixel_density": 1,
        "grid_size": (9,9),
        "targ_range": [1,5],
    }
    hyps["obs_shape"] = tuple(
        [hyps["n_frame_stack"],
        *hyps["grid_size"]]
    )
    data_collector = sg.experience.DataCollector(hyps)
    data_collector.dispatch_runners()
    data_collector.await_runners()

    exp = data_collector.exp_replay
    for k in exp.shared_exp.keys():
        print(k, "shape")
        print(exp.shared_exp[k].shape)
    for batch in range(hyps["batch_size"]//4):
        print("Viewing Batch", batch)
        for i in range(exp.shared_exp["obs"].shape[1]):
            print("done:", exp.shared_exp["dones"][batch][i].item())
            print("rew:", exp.shared_exp["rews"][batch][i].item())
            print("actn:", exp.shared_exp["actns"][batch][i].item())
            print()
            plt.imshow(exp.shared_exp["obs"][batch][i].numpy().squeeze())
            plt.show()
    data_collector.terminate_runners()

if __name__=="__main__":
    mp.set_start_method('forkserver')
    print("testing runner")
    test_runner()
    print("testing validation runner")
    test_validation_runner()
    print("testing data collector")
    test_data_collector()
