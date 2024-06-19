from .clip import *
from .ProDA import ProDACLIP
from .CoOp import CoOpCLIP
from .CoCoOp import CoCoOpCLIP
from .PLOT import PLOTCLIP
from .PromptSRC import PromptSRCCLIP
from .FedOTP import FedOTPCLIP
from .ProGrad import ProGradCLIP
from .BPLCLIP import BPLCLIP
from .KgCoOp import KgCoOpCLIP

clip_maps = {
    'CLIP': CoOpCLIP,
    'CoOp': CoOpCLIP,
    'CoCoOp': CoCoOpCLIP,
    'PLOT': PLOTCLIP,
    'ProDA': ProDACLIP,
    'PromptSRC': PromptSRCCLIP,
    'ProGrad': ProGradCLIP,
    'BPL': BPLCLIP,
    'KgCoOp': KgCoOpCLIP,
    'OTP': FedOTPCLIP,
}