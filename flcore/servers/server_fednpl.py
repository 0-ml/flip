from ..clients.client_fednpl import ClientFedNPL
from ..servers.server_fedavg import FedAvg
from statistics import harmonic_mean as hmean

from ..utils import DivergeError, eval_multiple
from ..pretty.logger import log


class FedNPL(FedAvg):
    def __init__(self, args, times):
        self.eval_multi = args.eval_multi == 'true'
        super().__init__(args, times)
        self.init_clients(ClientFedNPL)
        print(f"\nTrain fraction: {self.train_fraction}, total clients: {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.best_acc = 0.
        self.forward_times_list = range(2, 16, 2)
        self.best_multi_base = {t:0 for t in self.forward_times_list}
        self.best_multi_novel = {t:0 for t in self.forward_times_list}
        self.best_multi_hm = {t:0 for t in self.forward_times_list}

    def eval_multiple(self, ):
        if not self.eval_multi:
            return
        base_accs = eval_multiple(self.model, self.testloader_base, self.device,
                                 self.train_classnames, self.test_classnames_base,
                                 self.forward_times_list)
        novel_accs = eval_multiple(self.model, self.testloader_novel, self.device,
                                 self.train_classnames, self.test_classnames_novel,
                                 self.forward_times_list)
        self.tb.add_multiple_scalars(
            f'eval_multiple_base', base_accs, self.rounds)
        self.tb.add_multiple_scalars(
            f'eval_multiple_novel', novel_accs, self.rounds)

        for t in self.forward_times_list:
            base_acc, novel_acc = base_accs[t], novel_accs[t]
            self.best_multi_base[t] = base_acc if base_acc > self.best_multi_base[t]\
                                                     else self.best_multi_base[t]
            self.best_multi_novel[t] = novel_acc if novel_acc > self.best_multi_novel[t] \
                                                        else self.best_multi_novel[t]
            hm = hmean([base_acc, novel_acc])
            self.best_multi_hm[t] = hm if hm > self.best_multi_hm[t] \
                                                        else self.best_multi_hm[t]

        self.tb.add_multiple_scalars('eval_multiple_base/best', self.best_multi_base, self.rounds)
        self.tb.add_multiple_scalars('eval_multiple_novel/best', self.best_multi_novel, self.rounds)
        self.tb.add_multiple_scalars('eval_multiple_hm/best', self.best_multi_hm, self.rounds)
