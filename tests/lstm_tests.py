import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import supervised_gym as sg
from supervised_gym.models import *

if __name__ == "__main__":
    hyps = {
        "seed": 0,
        "h_size": 256,
        "bnorm": False,
        "n_frame_stack": 2,
        "lr": 0.0005,
        "l2": 0.001,
        "conv_noise": 0.1,
        "dense_noise": 0.1,

        "env_type": "gordongames-v0",
        "harsh": True,
        "pixel_density": 3,
        "grid_size": [15,15],
        "targ_range": [1,10],

        "batch_size": 64,
        "seq_len": 9,
        "exp_len": 1000,
        "n_val_samples": 10,
        "n_eval_eps": 10,
        "n_eval_steps": None,
        "randomize_order": True,

        "oracle_type": "GordonOracle",
        "optim_type": "Adam",
        "loss_fxn": "CrossEntropyLoss",
        "preprocessor": "null_prep",
        "best_by_key": "val_perc_correct_avg"
    }
    hyps["inpt_shape"] = tuple(
        [hyps["n_frame_stack"],
        *hyps["grid_size"]]
    )
    hyps["actn_size"] = 2
    data_len = 1000
    seq_len = hyps["seq_len"]
    batch_size = hyps["batch_size"]
    inpts = torch.rand(hyps["batch_size"], data_len, *hyps["inpt_shape"])
    targs = torch.randint(
        low=0,
        high=hyps["actn_size"],
        size=(hyps["batch_size"], data_len)
    )
    dones = torch.zeros(hyps["batch_size"], data_len).cuda()
    dones[:,49::50] = 1
    model = SimpleLSTM(**hyps)
    model.train()
    model.cuda()
    optm = getattr(torch.optim, hyps["optim_type"])(model.parameters(), lr=hyps["lr"])
    loss_fxn = getattr(torch.nn, hyps["loss_fxn"])()
    for epoch in range(100):
        print("Epoch:", epoch)
        model.reset(hyps["batch_size"])
        avg_acc = 0
        avg_loss = 0
        n_loops = 0
        for batch in range(0, data_len-seq_len, seq_len):
            n_loops += 1
            optm.zero_grad()
            x = inpts[:, batch:batch+seq_len].cuda()
            y = targs[:, batch:batch+seq_len].cuda().reshape(-1)
            preds = []
            preds = model(x, dones[:,batch:batch+seq_len].cuda())
            preds = preds.reshape(-1, preds.shape[-1])
            loss = loss_fxn(preds, y)
            loss.backward()
            optm.step()
            model.h = model.h.data
            model.c = model.c.data
            args = torch.argmax(preds, dim=-1)
            acc = (args==y).float().mean()
            avg_acc += acc.item()
            avg_loss += loss.item()
            print("Loss:", loss.item(), "- Acc:", acc.item(), "-", batch/data_len, "%", end="              \r")
        print("Avg Loss:", avg_loss/n_loops, "-- Avg Acc:", avg_acc/n_loops)
        print("\n")
