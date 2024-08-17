import argparse

from flcore.servers.server_fedavg import FedAvg
from flcore.servers.server_fedotp import FedOTP


server_maps = {
    'FedAvg': FedAvg,
    'FedOTP': FedOTP,
}
CUSTOM_SERVERS = {}


def register_server(name=None):
    def register_server_cls(cls):
        cname = name or cls.__name__
        if cname in CUSTOM_SERVERS:
            raise ValueError(f"Cannot register duplicate server ({cname})")
        CUSTOM_SERVERS[cname] = cls
        return cls
    return register_server_cls


def find_server_in_module(name):
    try:
        __import__(f'flcore.servers.server_{name.lower()}')
    except ImportError as e:
        raise ValueError(f"Server implementation {name} not found.") from e
    return CUSTOM_SERVERS[name]


def get_server(args, remaining_args, times):
    try:
        return server_maps[args.fed_algo](args, times)
    except KeyError:
        pass
    try:
        server_cls = find_server_in_module(args.fed_algo)
        if hasattr(server_cls, 'extra_arguments'):
            parser = argparse.ArgumentParser()
            for k, v in server_cls.extra_arguments.items():
                parser.add_argument(*k, **v)
            remaining_args = parser.parse_args(remaining_args)
            for k, v in vars(remaining_args).items():
                setattr(args, k, v)
        return server_cls(args, times)
    except KeyError:
        raise ValueError(f"Server {args.fed_algo} not found.")
