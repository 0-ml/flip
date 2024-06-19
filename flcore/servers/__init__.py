

from flcore.servers.server_fedavg import FedAvg
from flcore.servers.server_fedotp import FedOTP


server_maps = {
    'FedAvg': FedAvg,
    'FedOTP': FedOTP,
}


def get_server(args, times):
    server = server_maps[args.fed_algo](args, times)
    return server