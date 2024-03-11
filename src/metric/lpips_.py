from src.metric.base_metric import BaseMetric
from lpips import LPIPS
import torch


class LPIPS_Wrapper(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.loss_fn_alex = LPIPS(net="alex")

    def __call__(self, images, **kwargs):
        lpips_values = []
        for i in range(len(images) - 1):
            for j in range(i + 1, len(images)):
                lpips_values.append(self.loss_fn_alex(images[i], images[j]))
        lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
        return lpips_value.item()
