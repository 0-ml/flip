import copy
from .client_fedavg import ClientFedAvg

class ClientFedOTP(ClientFedAvg):
    len_history = 100

    def __init__(self, args, id, trainloader, testloader, **kwargs):
        super().__init__(args, id, trainloader, testloader, **kwargs)

    def keep_states(self, ):
        # ctx[0]: global avg ctx, ctx[1]: local ctx
        self.local_prompt = copy.deepcopy(self.model.prompt_learner.ctx[1])

    def load_states(self, ):
        self.model.prompt_learner.ctx[1].data = self.local_prompt