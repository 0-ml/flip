from ..clients.client_fedotp import ClientFedOTP
from ..servers.server_fedavg import FedAvg

from ..utils import DivergeError
from ..pretty.logger import log


class FedOTP(FedAvg):
    def __init__(self, args, times):
        args.prompt_algo = 'OTP'
        super().__init__(args, times)
        self.init_clients(ClientFedOTP)
        print(f"\nTrain fraction: {self.train_fraction}, total clients: {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.best_acc = 0.