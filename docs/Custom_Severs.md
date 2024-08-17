# Extending FLIP with Custom Server Implementations

## Overview

The FLIP benchmark is designed to be easily extensible
with custom server implementations.
This document describes how to extend FLIP
with your own server algorithms.

## Adding a Custom Server

To add a custom server implementation to FLIP,
add a new file named `server_<custom_name>.py`
to the `flcore/servers/` directory,
where `<custom_name>` is the lower-cased name of your custom server.
In this file,
define a class named `<CustomName>`
that extends `flcore.servers.FedAvg`.
[FedCustom](/flcore/servers/server_fedcustom.py)
is an example of a custom server implementation:
```python
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
```


## Registering the Custom Server

Note that the custom server class must be registered
using the `@register_server` decorator.
The `@register_server` decorator is a function
that takes an optional `name` argument
and registers the custom server class with the given name
to the server registry of the FLIP benchmark.
The `name` argument is optional
and defaults to the lower-cased class name of the custom server.
To use your custom server implementation,
run `main.py` with the `--fed_algo=<your_custom_name>` argument.


## Adding Custom Arguments

As shown in the example above,
you can add custom arguments to your custom server
using the `extra_arguments` class attribute.
The `extra_arguments` attribute is a dictionary
where the keys are tuples of argument names,
and the values are dictionaries
that specify the argument type, default value, and help message.
The custom arguments are automatically
added to the command-line interface
when you run the FLIP benchmark
with your custom server.
