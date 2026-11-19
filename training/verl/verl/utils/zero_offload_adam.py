"""ZeRO-Offload-style optimizer for the original Open-MOPD (verl FSDP2 CPUOffloadPolicy) path.

FSDP2's CPU offload policy already keeps the fp32 master parameters and gradients
on the host; ZeRO-Offload's contribution is the vectorized C++ CPU Adam. This is
DeepSpeedCPUAdam.step verbatim, except that FSDP2 DTensor parameters/gradients are
handed to the C++ kernel as their local tensors (single rank: the full tensor).
"""
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam


def _local(tensor):
    return tensor._local_tensor if hasattr(tensor, '_local_tensor') else tensor


class ZeroOffloadCPUAdam(DeepSpeedCPUAdam):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                data = _local(p.data)
                grad = _local(p.grad.data)
                assert data.device.type == 'cpu' and grad.device.type == 'cpu', (data.device, grad.device)
                assert data.dtype == torch.float32 and data.is_contiguous()
                if grad.dtype != torch.float32 or not grad.is_contiguous():
                    grad = grad.float().contiguous()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(data, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(data, dtype=torch.float32)
                state['step'] += 1
                beta1, beta2 = group['betas']
                self.ds_opt_adam.adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                             group['weight_decay'], group['bias_correction'], data, grad,
                                             state['exp_avg'], state['exp_avg_sq'])
        return loss
