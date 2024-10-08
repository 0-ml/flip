

from flcore.servers.server_fedavg import FedAvg
from flcore.servers.server_fedotp import FedOTP
from flcore.servers.server_central import Central


server_maps = {
    'FedAvg': FedAvg,
    'FedOTP': FedOTP,
    'Central': Central,
}


def get_server(args, times):
    server = server_maps[args.fed_algo](args, times)
    return server