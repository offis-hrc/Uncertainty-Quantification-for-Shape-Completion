import torch
import torch.nn as nn
import emd_cuda
 

class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        #print('CUDAA XYZ', xyz1)
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        #match = match.to('cuda:1')
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        #cost = cost.to('cuda:1')
        ctx.save_for_backward(xyz1, xyz2, match)
        ctx.mark_non_differentiable(match)
        #print('CUDAA MATCH', match)
        return cost, match

    @staticmethod
    def backward(ctx, grad_cost, grad_cost_2):
        xyz1, xyz2, match = ctx.saved_tensors
        
        grad_cost = grad_cost.contiguous()
        #print('CUDAA gardient', grad_cost)
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        #print('BACKKKK??', grad_xyz1)
        return grad_xyz1, grad_xyz2


class EarthMoverDistance(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, xyz1, xyz2):
        """
        Args:
            xyz1 (torch.Tensor): (b, N1, 3)
            xyz2 (torch.Tensor): (b, N2, 3)

        Returns:
            cost (torch.Tensor): (b)
        """
        if xyz1.dim() == 2:
            xyz1 = xyz1.unsqueeze(0)
        if xyz2.dim() == 2:
            xyz2 = xyz2.unsqueeze(0)
        cost, match = EarthMoverDistanceFunction.apply(xyz1, xyz2)
        return cost, match
