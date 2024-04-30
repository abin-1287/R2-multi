import torch
from torch.optim.lr_scheduler import LambdaLR
import math
import torch.optim as optim

def build_optimizer(args, model, criterion):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed},
        {'params': criterion.parameters(), 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, args, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.lr = args.lr_ed
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return self.lr_ed * (float(step) / float(max(1, self.warmup_steps)))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps))) * self.lr_ed

 
class WarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, args, optimizer, warmup_steps, last_epoch=-1):
        """
        optimizer: 优化器对象
        warmup_steps: 学习率线性增加的步数
        gamma: 学习率下降系数
        last_epoch: 当前训练轮数
        """
        self.warmup_steps = warmup_steps
        self.gamma = args.gamma
        super().__init__(optimizer, last_epoch)
 
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 学习率线性增加
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # 学习率按指数衰减
            return [base_lr * math.exp(-(self.last_epoch - self.warmup_steps + 1) * self.gamma) for base_lr in self.base_lrs]
 
