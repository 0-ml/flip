

from flcore.servers.server_fedavg import FedAvg
from flcore.servers.server_fedotp import FedOTP
from flcore.servers.server_central import Central
from flcore.servers.server_fedpgp import FedPGP
from flcore.servers.server_maple import MaPLe
from flcore.servers.server_folio import FedFolio
from flcore.servers.server_dpfpl import FedDPFPL


server_maps = {
    'FedAvg': FedAvg,
    'FedOTP': FedOTP,
    'Central': Central,
    'FedPGP': FedPGP,
    'MaPLe': MaPLe,
    'Folio': FedFolio,
    'FedDPFPL': FedDPFPL,
}


def get_server(args, times):
    server = server_maps[args.fed_algo](args, times)
    return server