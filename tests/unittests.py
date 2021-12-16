import unittest
import supervised_gym as sg
import torch
import numpy as np
import torch.multiprocessing as mp

class ExperienceTests(unittest.TestCase):
    def test_shared_tensors(self):
        hyps = {
            "exp_len": 13,
            "batch_size": 16,
            "obs_shape": (3,9,9),
            "seq_len": 7,
            "randomize_order": False,
        }
        exp = sg.experience.ExperienceReplay(**hyps)
        self.assertEqual(
            exp.shared_exp["obs"].shape,
            tuple([
                hyps["batch_size"],
                hyps["exp_len"],
                *hyps['obs_shape']
            ])
        )
        self.assertEqual(
            exp.shared_exp["rews"].shape,
            tuple([
                hyps["batch_size"],
                hyps["exp_len"],
            ])
        )
        self.assertEqual(
            exp.shared_exp["dones"].shape,
            tuple([
                hyps["batch_size"],
                hyps["exp_len"],
            ])
        )
        self.assertEqual(
            exp.shared_exp["actns"].shape,
            tuple([
                hyps["batch_size"],
                hyps["exp_len"],
            ])
        )

    def test_len(self):
        hyps = {
            "exp_len": 13,
            "batch_size": 16,
            "obs_shape": (3,9,9),
            "seq_len": 7,
            "randomize_order": False,
        }
        exp = sg.experience.ExperienceReplay(**hyps)
        self.assertEqual(len(exp), hyps["exp_len"]-hyps['seq_len'])
        hyps = {
            "exp_len": 5,
            "batch_size": 16,
            "obs_shape": (3,9,9),
            "seq_len": 7,
            "randomize_order": False,
        }
        try:
            exp = sg.experience.ExperienceReplay(**hyps)
            self.assertFalse(True)
        except Exception as e:
            self.assertEqual(type(e), AssertionError)

    def test_iter(self):
        hyps = {
            "exp_len": 13,
            "batch_size": 16,
            "obs_shape": (3,9,9),
            "seq_len": 7,
            "randomize_order": False,
        }
        exp = sg.experience.ExperienceReplay(**hyps)
        for k in exp.shared_exp.keys():
            numel = exp.shared_exp[k].numel()
            ar = torch.arange(numel).type(exp.shared_exp[k].dtype)
            exp.shared_exp[k][:] = ar.reshape(exp.shared_exp[k].shape)

        loop_count = 0
        old_data = None
        for i,data in enumerate(exp):
            self.assertEqual(
                data["obs"].shape,
                tuple([
                    hyps["batch_size"],
                    hyps["seq_len"],
                    *hyps['obs_shape']
                ])
            )
            self.assertEqual(
                data["rews"].shape,
                tuple([
                    hyps["batch_size"],
                    hyps["seq_len"],
                ])
            )
            self.assertEqual(
                data["dones"].shape,
                tuple([
                    hyps["batch_size"],
                    hyps["seq_len"],
                ])
            )
            self.assertEqual(
                data["actns"].shape,
                tuple([
                    hyps["batch_size"],
                    hyps["seq_len"],
                ])
            )
            if old_data is None:
                old_data = {**data}
            else:
                for k in old_data.keys():
                    self.assertFalse(
                        np.array_equal(
                            old_data[k],
                            data[k]
                        )
                    )

class SimpleCNNTests(unittest.TestCase):
    def test_feat_std(self):
        hyps = {
            "inpt_shape": (1,9,9),
            "actn_size": 5,
            "h_size": 128,
            "bnorm": True,
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
        model = sg.models.SimpleCNN(**hyps)
        inpt = torch.randn(3,*hyps["inpt_shape"])

        feats = model.features(inpt)
        for i in range(len(inpt)):
            self.assertTrue(
                feats[i].std() < 3 and feats[i].std() > .1
            )

    def test_feat_inf(self):
        hyps = {
            "inpt_shape": (1,9,9),
            "actn_size": 5,
            "h_size": 128,
            "bnorm": True,
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
        model = sg.models.SimpleCNN(**hyps)
        inpt = torch.randn(3,*hyps["inpt_shape"])
        feats = model.features(inpt)
        inf_exists = False
        for el in feats.flatten():
            if el.item() == np.inf:
                inf_exists = True
                break
        self.assertFalse(inf_exists)

    def test_feat_nan(self):
        hyps = {
            "inpt_shape": (1,9,9),
            "actn_size": 5,
            "h_size": 128,
            "bnorm": True,
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
        model = sg.models.SimpleCNN(**hyps)
        inpt = torch.randn(3,*hyps["inpt_shape"])
        feats = model.features(inpt)
        nan_exists = False
        for el in feats.flatten():
            if el.item() != el.item():
                nan_exists = True
                break
        self.assertFalse(nan_exists)

    def test_dense(self):
        hyps = {
            "inpt_shape": (1,9,9),
            "actn_size": 5,
            "h_size": 128,
            "bnorm": True,
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
        model = sg.models.SimpleCNN(**hyps)
        inpt = torch.randn(3,*hyps["inpt_shape"])
        feats = model.features(inpt)
        actns = model.dense(feats)

        temp = model(inpt[None])
        self.assertTrue(
            np.array_equal(
                temp.data.squeeze().numpy(),
                actns.data.numpy()
            )
        )

    def test_dense_std(self):
        hyps = {
            "inpt_shape": (1,9,9),
            "actn_size": 5,
            "h_size": 128,
            "bnorm": True,
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
        model = sg.models.SimpleCNN(**hyps)
        inpt = torch.randn(1,3,*hyps["inpt_shape"])
        actns = model(inpt)

        self.assertTrue(
            actns.std() < 3 and actns.std() > .1
        )

    def test_dense_inf(self):
        hyps = {
            "inpt_shape": (1,9,9),
            "actn_size": 5,
            "h_size": 128,
            "bnorm": True,
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
        model = sg.models.SimpleCNN(**hyps)
        inpt = torch.randn(1,3,*hyps["inpt_shape"])
        actns = model(inpt)

        inf_exists = False
        for el in actns.flatten():
            if el.item() == np.inf:
                inf_exists = True
                break
        self.assertFalse(inf_exists)

    def test_dense_nan(self):
        hyps = {
            "inpt_shape": (1,9,9),
            "actn_size": 5,
            "h_size": 128,
            "bnorm": True,
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
        model = sg.models.SimpleCNN(**hyps)
        inpt = torch.randn(1,3,*hyps["inpt_shape"])
        actns = model(inpt)

        nan_exists = False
        for el in actns.flatten():
            if el.item() != el.item():
                nan_exists = True
                break
        self.assertFalse(nan_exists)

if __name__=="__main__":
    unittest.main()
