import copy
from .client_fedavg import ClientFedAvg

class ClientFedPGP(ClientFedAvg):
    len_history = 100

    def __init__(self, args, id, trainloader, testloader, **kwargs):
        super().__init__(args, id, trainloader, testloader, **kwargs)

    def keep_states(self, ):
        # ctx[0]: global avg ctx, ctx[1]: local ctx
        self.local_prompt_U = copy.deepcopy(self.model.prompt_learner.U)
        self.local_prompt_V = copy.deepcopy(self.model.prompt_learner.V)

    def load_states(self, ):
        self.model.prompt_learner.U.data = self.local_prompt_U
        self.model.prompt_learner.V.data = self.local_prompt_V