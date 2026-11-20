"""SuperPoint with access to the dense descriptor map, on cuda / mps / cpu.

The LightGlue SuperPoint weights are reused; only the forward pass is split so the descriptor map can be sampled
at arbitrary points (needed to attach descriptors to the existing SfM observations at several scales).
On this Intel MacBook (AMD GPU via MPS) a 640x360 frame takes ~45 ms instead of ~650 ms on the CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _nms(scores: torch.Tensor, r: int) -> torch.Tensor:
    from lightglue.superpoint import simple_nms
    return simple_nms(scores[None], r)[0]


class SuperPointDense:
    def __init__(self, device: str = "auto", nms_radius: int = 4, threshold: float = 0.0005, border: int = 4):
        from lightglue import SuperPoint
        self.device = pick_device(device)
        self.net = SuperPoint().eval().to(self.device)
        self.nms_radius, self.threshold, self.border = nms_radius, threshold, border

    @torch.inference_mode()
    def dense(self, gray: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """gray uint8 (H, W) -> scores (H8, W8) and L2-normalised descriptors (256, H8/8, W8/8), on device.
        The image is cropped to a multiple of 8 (at most 7 px lost on the right/bottom)."""
        h, w = gray.shape[0] // 8 * 8, gray.shape[1] // 8 * 8
        x = torch.from_numpy(np.ascontiguousarray(gray[:h, :w])).to(self.device).float().div_(255)[None, None]
        n = self.net
        x = n.pool(n.relu(n.conv1b(n.relu(n.conv1a(x)))))
        x = n.pool(n.relu(n.conv2b(n.relu(n.conv2a(x)))))
        x = n.pool(n.relu(n.conv3b(n.relu(n.conv3a(x)))))
        x = n.relu(n.conv4b(n.relu(n.conv4a(x))))
        s = F.softmax(n.convPb(n.relu(n.convPa(x))), 1)[:, :-1]
        _, _, hc, wc = s.shape
        s = s.permute(0, 2, 3, 1).reshape(hc, wc, 8, 8).permute(0, 2, 1, 3).reshape(hc * 8, wc * 8)
        d = F.normalize(n.convDb(n.relu(n.convDa(x))), p=2, dim=1)[0]
        return s, d

    @staticmethod
    def sample(desc: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """Bilinear descriptors at pixel positions pts (N, 2) (pixel centres at integers). Returns (N, 256)."""
        c, h, w = desc.shape
        if len(pts) == 0:                      # MPS grid_sample asserts on an empty grid
            return torch.zeros(0, c, device=desc.device, dtype=desc.dtype)
        p = pts - 4 + 0.5
        p = p / torch.tensor([w * 8 - 4 - 0.5, h * 8 - 4 - 0.5], device=desc.device, dtype=p.dtype)
        p = p * 2 - 1
        out = F.grid_sample(desc[None], p.view(1, 1, -1, 2).float(), mode="bilinear", align_corners=True)
        return F.normalize(out.view(c, -1), p=2, dim=0).T

    @torch.inference_mode()
    def detect(self, gray: np.ndarray, max_kp: int = 1024, mask: np.ndarray | None = None):
        """Keypoints (N, 2) float32 numpy, scores (N,), descriptors (N, 256) torch tensor on device."""
        s, d = self.dense(gray)
        s = _nms(s, self.nms_radius)
        b = self.border
        s[:b], s[-b:], s[:, :b], s[:, -b:] = -1, -1, -1, -1
        if mask is not None:
            m = torch.from_numpy(np.ascontiguousarray(mask[:s.shape[0], :s.shape[1]] > 0)).to(s.device)
            s = torch.where(m, s, torch.full_like(s, -1))
        flat = s.reshape(-1)
        k = min(max_kp, int((flat > self.threshold).sum().item()))
        if k == 0:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.float32), torch.zeros(0, 256, device=s.device)
        sc, idx = torch.topk(flat, k)
        w = s.shape[1]
        kp = torch.stack([idx % w, idx // w], 1).float()
        desc = self.sample(d, kp)
        return kp.cpu().numpy(), sc.cpu().numpy(), desc
