import torch.nn as nn
import torch
import torch.nn.functional as F
from srcs.model.bd_model import BDNeRV, BDNeRV_RC, BDNeRV_RC_noTEM
from srcs.model.ce_model import CEBlurNet

class CEBDNet(nn.Module):
    '''
    coded exposure blur decomposition network
    '''
    def __init__(self, sigma_range=0, test_sigma_range=None, ce_code_n=8, frame_n=8, ce_code_init=None, opt_cecode=False, ce_net=None, binary_fc=None, bd_net=None):
        super(CEBDNet, self).__init__()
        self.ce_code_n = ce_code_n
        self.frame_n = frame_n
        # coded exposure blur net
        if ce_net == 'CEBlurNet':
            self.BlurNet = CEBlurNet(
                sigma_range=sigma_range, test_sigma_range=test_sigma_range, ce_code_n=ce_code_n, frame_n=frame_n, ce_code_init=ce_code_init, opt_cecode=opt_cecode, binary_fc=binary_fc)
        else:
            raise NotImplementedError(f'No model named {ce_net}')

        # deep deblur net
        if bd_net == 'BDNeRV':
            self.DeBlurNet = BDNeRV()
        elif bd_net=='BDNeRV_RC':
            self.DeBlurNet = BDNeRV_RC()
        # elif bd_net == 'BDNeRV_BRC':
        #     self.DeBlurNet = BDNeRV_BRC()
        elif bd_net == 'BDNeRV_RC_noTEM':
            self.DeBlurNet = BDNeRV_RC_noTEM()
        else:
            raise NotImplementedError(f'No model named {bd_net}')

    def forward(self, frames):
        ce_blur_img_noisy, time_idx, ce_code_up, ce_blur_img = self.BlurNet(
            frames)
        output = self.DeBlurNet(ce_blur_img_noisy, time_idx, ce_code_up)
        return output, ce_blur_img, ce_blur_img_noisy
