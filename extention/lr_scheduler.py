from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

# milestones:[50000, 70000]
# WARM_UP_FACTOR: float = 0.3333
# WARM_UP_NUM_ITERS: int = 500

class WarmUpMultiStepLR(MultiStepLR):
    def __init__(self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1,
                 factor: float = 0.3333, num_iters: int = 500, last_epoch: int = -1):
        self.factor = factor # warm-up factor
        self.num_iters = num_iters # warm-up itets
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.num_iters and (self.last_epoch == 199 or self.last_epoch==299):
            factor = 0.5
        else:
            factor = 1

        lr_l=[lr * factor for lr in super().get_lr()]
        return lr_l


        # if self.last_epoch < self.num_iters and (self.last_epoch+1) % 100==0:
        #     alpha = self.last_epoch / self.num_iters
        #     factor = (1 - self.factor) * alpha + self.factor
        # else:
        #     factor = 1

        # lr_l=[lr * factor for lr in super().get_lr()]
        # #print(lr_l)
        # return lr_l 
        # #return  super().get_lr()
