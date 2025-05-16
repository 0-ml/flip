import math
import torch

from ..clients.client_dpfpl import ClientDPFPL
from ..servers.server_fedavg import FedAvg

from ..utils import DivergeError
from ..pretty.logger import log




class FedDPFPL(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.init_clients(ClientDPFPL)
        print(f"\nTrain fraction: {self.train_fraction}, total clients: {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.best_acc = 0.
