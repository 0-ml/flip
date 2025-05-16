from ..clients.client_folio import ClientFolio
from ..servers.server_fedavg import FedAvg

from ..utils import DivergeError
from ..pretty.logger import log


class FedFolio(FedAvg):
    def __init__(self, args, times):
        args.prompt_algo = 'Folio'
        super().__init__(args, times)
        self.init_clients(ClientFolio)
        print(f"\nTrain fraction: {self.train_fraction}, total clients: {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.best_acc = 0.