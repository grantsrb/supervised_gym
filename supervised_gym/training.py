from datas import ExperienceReplay, DataCollector
from models import Model
from torch.optim import Adam, RMSprop


def train(rank, hyps, verbose=False):
    """
    This is the main training function. Argue a set of hyperparameters
    and this function will train a model to solve an openai gym task
    given an AI oracle.

    Args:
        rank: int
            the index of the distributed training system.
        hyps: dict
            a dict of hyperparams
            keys: str
            vals: object
        verbose: bool
            determines if the function should print status updates
    """
    # Initialize Experience Replay
    exp_replay = ExperienceReplay(hyps)
    # Initialize Data Collector
    data_collector = DataCollector(hyps)
    # Initialize model
    model = Model(hyps)
    # initialize trainer
    trainer = Trainer(hyps, model)
    # Loop training
    for epoch in range(hyps["n_epochs"]):
        if verbose:
            print()
            print("Starting Epoch", epoch)
        # Run environments
        new_data = data_collector.collect()
        exp_replay.add_data(new_data)
        trainer.train(model, exp_replay, hyps)

class Trainer:
    """
    This class handles the training of the model.
    """
    def __init__(self, hyps, model):
        """
        Args:
            hyps: dict
                keys: str
                vals: object
            model: torch.Module
        """
        self.hyps = hyps
        self.optim = self.get_optimizer(model, **hyps)

    def get_optimizer(self, model, optim_type, lr, *args, **kwargs):
        """
        Initializes an optimizer using the model parameters and the
        hyperparameters.
    
        Args:
            model: Model or torch.Module
                any object that implements a `.parameters()` member
                function that returns a sequence of torch.Parameters
            optim_type: str (one of [Adam, RMSprop])
                the type of optimizer. 
            lr: float
                the learning rate
        Returns:
            optim: torch optimizer
                the model optimizer
        """
        return globals()[optim_type](model.parameters(), lr=lr)

    def train(self, model, exp_replay):
        """
        This function handles the actual training. It loops through the
        available data from the experience replay to train the model.

        Args:
            model: torch.Module
                the model to be trained
            exp_replay: ExperienceReplay
                the collected experience/data. must implement __iter__
                and __len__ so that the data can be easily looped
                through.
        """
        for i,(x,y) in enumerate(exp_replay):
            preds = model(x)
            loss = self.calc_loss(preds, y)
            loss.backward()
            optim.step()

