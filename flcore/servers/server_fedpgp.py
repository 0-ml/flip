from ..clients.client_fedpgp import ClientFedPGP
from ..servers.server_fedavg import FedAvg

from ..utils import DivergeError
from ..pretty.logger import log


class FedPGP(FedAvg):
    """ Harmonizing Generalization and Personalization in Federated Prompt Learning
    """
    def __init__(self, args, times):
        args.prompt_algo = 'PGP'
        super().__init__(args, times)
        self.init_clients(ClientFedPGP)
        print(f"\nTrain fraction: {self.train_fraction}, total clients: {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.best_acc = 0.