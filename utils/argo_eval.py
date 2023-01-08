import torch
from av2.datasets.motion_forecasting.eval.metrics import compute_ade, compute_fde, compute_brier_fde


class ArgoEval:
    def __init__(self):
        pass

    def reshape(self, pred, y):
        pred = pred.reshape(-1, 60, 5).numpy()
        pred = pred[..., :2]
        y = y.reshape(60, 5)
        y = y[..., :2].numpy()
        return pred, y

    def forward(self, pred, y):
        pred, y = self.reshape(pred, y)

        # Input: (K, N, 2), (N, 2) -> Output: (K,)
        ade_k = compute_ade(pred, y)
        # Input: (K, N, 2), (N, 2) -> Output: (K,)
        fde_k = compute_fde(pred, y)

        min_ade = ade_k.min()
        min_fde = fde_k.min()

        eval_result = {
            'min_ade': min_ade,
            'min_fde': min_fde,
        }
        return eval_result
