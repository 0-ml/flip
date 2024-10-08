from .clip import *
from .ZSCLIP import ZSCLIP
from .ProDA import ProDACLIP
from .CoOp import CoOpCLIP
from .CoCoOp import CoCoOpCLIP
from .PLOT import PLOTCLIP
from .PromptSRC import PromptSRCCLIP
from .FedOTP import FedOTPCLIP
from .ProGrad import ProGradCLIP
from .BPLCLIP import BPLCLIP
from .KgCoOp import KgCoOpCLIP

from .CoOp import DenseCoOpCLIP

clip_maps = {
    'CLIP': ZSCLIP,
    'CoOp': CoOpCLIP,
    'CoCoOp': CoCoOpCLIP,
    'PLOT': PLOTCLIP,
    'ProDA': ProDACLIP,
    'PromptSRC': PromptSRCCLIP,
    'ProGrad': ProGradCLIP,
    'BPL': BPLCLIP,
    'KgCoOp': KgCoOpCLIP,
    'OTP': FedOTPCLIP,
    'DenseCoOp': DenseCoOpCLIP,
}