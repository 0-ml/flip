import time
import torch
import numpy as np
import torch.nn as nn
from statistics import harmonic_mean as hmean
from matplotlib import pyplot as plt

from ..clients.client_fedavg import ClientFedAvg
from ..servers.server_base import ServerBase
from ..utils import ( DivergeError, eval_global, eval_base_novel,
                     eval_personal, svd, eval_domains, collect_state,
                     load_state_dict)
from ..pretty.logger import log
# from scipy.spatial.distance import cosine as CosineSimilarity
from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity

class FedAvg(ServerBase):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.init_clients(ClientFedAvg)
        print(f"\nTrain fraction: {self.train_fraction}, total clients: {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.best_acc = 0.
        self.best_acc_novel = 0.
        self.best_acc_hm = 0.
        self.best_acc_per = 0.
        self.best_acc_mds = {}
        self.id_text_feats = {}
        self.ood_text_feats = {}

    def run(self):
        if self.prompt_algo == 'CLIP':
            if self.bench == 'multidomain':
                top1, md_accs = eval_domains(self.model, self.testloaders_mdomain,
                                   self.device, precision=self.precision)
                if top1 > self.best_acc:
                    self.best_acc = top1
                    self.best_acc_mds = md_accs
            else:
                top1, _ = eval_global(self.model, self.testloader, self.device,
                                      self.precision, self.task)
                self.best_acc = top1
                self.best_acc_per = top1
                self.best_acc_hm = top1
            log.info(f'CLIP acc: {top1:.2%}')
        else:
            log.info('Starting Local Training...')
            for i in range(self.global_rounds+1):
                self.rounds = i
                start_time = time.time()
                self.selected_client_ids = self.select_clients()
                info, top1 = self.comm_round()
                log.info(f'{info}, top1: {top1:.2%}, '
                         f'elapsed: {time.time() - start_time:.1f}s, '
                         f'best_acc: {self.best_acc:.2%}.')

        self.summarize()


    def comm_round(self, ):
        results = {}
        top1 = 0.

        for c in self.selected_client_ids:
            self.clients[c].init(self.model, self.rounds)
            results[c] = self.clients[c].local_train()
            # self.check_clip(c)

        if not results:
            raise DivergeError('All clients trained to divergence.')

        states, weights, info = self.receive_results(results)
        avg_state = self.aggregate(states, weights)
        self.model = load_state_dict(self.model, avg_state, self.task)
        self.model.custom_avg(self.rounds)
        avg_state = collect_state(self.model, self.task)
        if self.eval_rounds > 1 and (self.rounds % self.eval_rounds != 0):
            return info, top1

        if self.bench == 'base2novel':
            top1, _ = eval_base_novel(self.model, self.testloader_base, self.device,
                                 self.train_classnames, self.test_classnames_base, self.precision)
            top1_novel, _ = eval_base_novel(self.model, self.testloader_novel, self.device,
                                 self.train_classnames, self.test_classnames_novel, self.precision)
            top1_hm = hmean([top1, top1_novel])
            if top1_hm > self.best_acc_hm:
                self.best_acc = top1
                self.best_acc_novel = top1_novel
                self.best_acc_hm = top1_hm
            info += f', eval_novel: {top1_novel:.2%}'
            info += f', best_hm: {self.best_acc_hm:.2%}'
            self.tb.add_scalar('eval/top1_base', top1, self.rounds)
            self.tb.add_scalar('eval/best_base', self.best_acc, self.rounds)
            self.tb.add_scalar('eval/top1_novel', top1_novel, self.rounds)
            self.tb.add_scalar('eval/best_novel', self.best_acc_novel, self.rounds)
            self.tb.add_scalar('eval/top1_hmean', top1_hm, self.rounds)
            self.tb.add_scalar('eval/best_hmean', self.best_acc_hm, self.rounds)
            self.eval_multiple()
        elif self.bench == 'dual':
            top1, _ = eval_global(self.model, self.testloader, self.device,
                                  self.precision, self.task)
            top1_per, _ = eval_personal(self.model, states, weights,
                                        self.testloaders_per, self.device, self.precision)
            self.model.prompt_learner.load_state_dict(avg_state, strict=False) # reload avg_state
            top1_hm = hmean([top1, top1_per])
            if top1_hm > self.best_acc_hm:
                self.best_acc = top1
                self.best_acc_per = top1_per
                self.best_acc_hm = top1_hm
            info += f', eval_per: {top1_per:.2%}'
            info += f', best_hm: {self.best_acc_hm:.2%}'
            self.tb.add_scalar('eval/top1_global', top1, self.rounds)
            self.tb.add_scalar('eval/best_global', self.best_acc, self.rounds)
            self.tb.add_scalar('eval/top1_personal', top1_per, self.rounds)
            self.tb.add_scalar('eval/best_personal', self.best_acc_per, self.rounds)
            self.tb.add_scalar('eval/top1_hmean', top1_hm, self.rounds)
            self.tb.add_scalar('eval/best_hmean', self.best_acc_hm, self.rounds)
        elif self.bench == 'personal':
            top1, _ = eval_personal(self.model, states, weights, self.testloaders_per,
                                        self.device, self.precision)
            self.model.prompt_learner.load_state_dict(avg_state, strict=False) # reload avg_state
            self.best_acc = top1 if top1 > self.best_acc else self.best_acc
            self.best_acc_per = top1 if top1 > self.best_acc_per else self.best_acc_per
            self.tb.add_scalar('eval/top1_personal', top1, self.rounds)
            self.tb.add_scalar('eval/best_personal', self.best_acc, self.rounds)
        elif self.bench == 'multidomain':
            top1, md_accs = eval_domains(self.model, self.testloaders_mdomain,
                                   self.device, precision=self.precision)
            if top1 > self.best_acc:
                self.best_acc = top1
                self.best_acc_mds = md_accs
            self.tb.add_scalar('eval/top1_domain_weighted', top1, self.rounds)
            self.tb.add_scalar('eval/best_domain_weighted', self.best_acc, self.rounds)
            self.tb.add_multiple_scalars('eval/multidomain', self.best_acc_mds, self.rounds)
        else:
            # global, xdomain
            top1, _ = eval_global(self.model, self.testloader, self.device,
                                  self.precision, self.task)
            self.best_acc = top1 if top1 > self.best_acc else self.best_acc
            self.tb.add_scalar('eval/top1', top1, self.rounds)
            self.tb.add_scalar('eval/best', self.best_acc, self.rounds)
        return info, top1

    def check_clip(self, c):
        for (n1, p1) in self.model.named_parameters():
            print(f'layer name: {n1}')
        for (n1, p1), (n2, p2) in zip(self.model.named_parameters(),
                                      self.clients[c].model.named_parameters()):
            print(f'layer name: {n1}')
            delta = 1e-8 * torch.ones_like(p1).to(self.device)
            diff = torch.any(torch.gt((p1 - p2), delta))
            if diff:
                print('weight difference!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
