import pandas as pd
import supervised_gym.utils.save_io as sgio

def get_stats_dataframe(model_folders, names=None):
    """
    Sorts through all checkpoints of all models contained within the
    model_folders array.  Returns a dataframe with all of their stats.
    Includes the following columns for ease of use:

        epoch: the epoch of the row in the dataframe
        model_path: the full path to the model folder
        model_num: the number associated with this model (the number
            surrounded by underscores following the experiment name)
        name: a name for the model. if names is not none, will draw
            from the corresponding entry in that array. Otherwise
            defaults to "model0", "model1", ...

    ASSUMES ALL CHECKPOINTS HAVE A "STATS" KEY THAT CONTAINS THE SAME
    KEYS OVER ALL CHECKPOINTS OF ALL MODELS

    Args:
        model_folders: list of str
            the full paths to each of the model experiments to be
            included in the dataframe
        names: list of str or None
            optional argument to name each model. If not None, the
            array must be as long as the model_folders array. if None,
            each name defaults to "model<int>" where <int> is repaced
            by the order in which this model was processed in the loop
    """
    extras = ["epoch", "model_path", "model_num", "name"]
    if len(model_folders) == 0:
        return pd.DataFrame({ k: [] for k in extras })
    if names is None: 
        names = ["model"+str(i) for i in range(len(model_folders))]
    assert len(names) == len(model_folders)
    
    checkpt = sgio.load_checkpoint(model_folders[0])
    stats = checkpt["stats"]
    keys = [k for k in stats.keys()]
    df = {k:[] for k in keys+extras}
    main_df = pd.DataFrame(df)
    
    for model_folder, name in zip(model_folders, names):
        checkpts = sgio.get_checkpoints(model_folder)
        df = {k:[] for k in keys}
        for path in checkpts:
            checkpt = sgio.load_checkpoint(path)
            for k in checkpt["stats"].keys():
                df[k].append(checkpt["stats"][k])
        df = pd.DataFrame(df)
        df["epoch"] = list(range(len(df)))
        df["model_path"] = model_folder
        df["name"] = name
        df["model_num"] = checkpt["hyps"]["exp_num"]
        main_df = main_df.append(df)
    return main_df
