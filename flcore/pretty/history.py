import os
import json

import torch
from torch.utils.tensorboard import summary, SummaryWriter


class History(SummaryWriter):
    def __init__(self, name):
        from .logger import log  # pylint: disable=import-outside-toplevel
        out_dir = os.path.join('summaries', name)
        if os.path.exists(out_dir):
            action = 'Resuming'
        else:
            action = 'Creating'
            os.makedirs(out_dir)
        log.verbose(f'{action} tensorboard summaries at {out_dir!r}...')
        super().__init__(out_dir, flush_secs=5)
        self.history_name = os.path.join(out_dir, 'history')
        # pylint: disable=consider-using-with
        self.history_file = open(self.history_name, 'w+')

    def flush(self):
        self.history_file.flush()
        return super().flush()

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if not isinstance(scalar_value, torch.Tensor):
            value = scalar_value
        elif scalar_value.dtype in [torch.float16, torch.float32, torch.float64]:
            value = float(scalar_value)
        elif scalar_value.dtype in [torch.int16, torch.int32, torch.int64]:
            value = int(scalar_value)
        else:
            raise TypeError('Unknown dtype.')
        line = json.dumps({'step': global_step, tag: value})
        self.history_file.write(f'{line}\n')
        return super().add_scalar(
            tag, scalar_value, global_step=global_step, walltime=walltime)

    def add_multiple_scalars(
            self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        for k, v in tag_scalar_dict.items():
            if v is None:
                continue
            self.add_scalar(f'{main_tag}/{k}', v, global_step, walltime)

    def add_hparams(self, hparam_dict, metric_dict):
        if any(not isinstance(d, dict) for d in (hparam_dict, metric_dict)):
            raise TypeError(
                'hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = summary.hparams(hparam_dict, metric_dict)
        logdir = self._get_file_writer().get_logdir()
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)
