import torch
import torch.distributed as dist
from src.train.loss.loss import loss
import matplotlib.pyplot as plt
import seaborn as sb
import torch.nn.functional as F
class loss_0(loss):
    def __init__(
        self,
        idx=0,
        name='Barlow Twins Loss',
        description='Cross correlation matrix based loss from the barlow twins paper ',
        input_='B x D',
        output='1D',
        lambda_param=0.005,
    ) -> None:
        super().__init__(idx, name, description, input_, output)
        self.lambda_param = lambda_param

    def forward(self,
                out_1=None,
                out_2=None,
                epoch=0,
                device=None,
                world_size=1):
        out_1 = F.normalize(out_1,p=2,dim=0)
        print(out_1.mean(), out_1.std())
        batch_size = out_1.size(0)
        D = out_1.size(1)
        # cross-correlation matrix
        c = torch.mm(out_1.T, out_1) / (batch_size * world_size)
        print(torch.diag(c))
        quit()
        if world_size > 1:
            dist.all_reduce(c)
        if True or epoch % 100 == 0:
            ans = sb.heatmap(c.cpu().data.numpy(), cmap="Blues")
            plt.show()
        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss
