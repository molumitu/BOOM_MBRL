class Trainer:
    """Base trainer class."""

    def __init__(self, cfg, env, agent, buffer, logger):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        print("Architecture:", self.agent.model)
        # print("Learnable parameters: {:,}".format(self.agent.model.total_params))

    def eval(self):
        """Evaluate a agent."""
        raise NotImplementedError

    def train(self):
        """Train a agent."""
        raise NotImplementedError
