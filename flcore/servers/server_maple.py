from ..clients.client_fedavg import ClientFedAvg
from ..servers.server_fedavg import FedAvg

from ..utils import DivergeError
from ..pretty.logger import log


class MaPLe(FedAvg):
    def __init__(self, args, times):
        self.args = args
        args.prompt_algo = 'MaPLe'
        super().__init__(args, times)
        self.init_clients(ClientFedAvg)
        print(f"\nMaPLe: Train fraction: {self.train_fraction}, total clients: {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.best_acc = 0.

    def gen_design_details(self,):
        design_details = {"trainer": 'MaPLe',
                          "vision_depth": 0,
                          "language_depth": 0,
                          "vision_ctx": 0,
                          "language_ctx": 0,
                          "maple_length": self.args.num_context}
        return design_details
