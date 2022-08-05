from src.train.optimizer.optimizer import optimizer
import torch
from torch.nn.parameter import Parameter

class optimizer_0(optimizer):
    def __init__(self,
                idx = 0, 
                params = Parameter(torch.randn(2, 2, requires_grad=True)),
                lr = 0.001, 
                weight_decay = 0, 
                momentum = 0.9, 
                eta = 0.001,
                weight_decay_filter = None,
                lars_adaptation_filter = None,
                betas =(0.9, 0.999),
                name = 'LARS',
                description = 'LARS optimiser implementation'):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults, idx, name, description)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad
                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,torch.where(update_norm > 0,(g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)
                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])
        