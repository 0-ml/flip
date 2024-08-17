from . import register_server
from .server_fedavg import FedAvg


@register_server()
class FedCustom(FedAvg):
    extra_arguments = {
        ('--custom', ): {
            'type': str, 'default': 'x',
            'help': 'Custom argument for custom server.',
        },
    }

    def __init__(self, args, times):
        super().__init__(args, times)
        custom_args = self._assign_arguments(args)
        print(f'Custom server with custom arguments: {custom_args}')

    def aggregate(self, states, weights):
        # Write your custom aggregation algorithm here...
        return super().aggregate(states, weights)
