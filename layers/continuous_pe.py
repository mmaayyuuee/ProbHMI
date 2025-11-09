import torch
import torch.nn as nn
import math


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    if len(timesteps.shape) < 1:
        timesteps = torch.reshape(timesteps, shape=(-1,))
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    # scale embeddings
    emb = scale * emb
    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ContinuousTimesteps(nn.Module):
    def __init__(self, num_channels:int, flip_sin_to_cos:bool, downscale_freq_shift:float, scale:int=1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        return

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class ContinuousTimestepEmbedder(nn.Module):
    def __init__(self, num_channels, CTS):
        super().__init__()
        self.num_channels = num_channels
        self.CTS = CTS
        self.time_embed = nn.Sequential(
            nn.Linear(self.num_channels, self.num_channels),
            nn.SiLU(),
            nn.Linear(self.num_channels, self.num_channels),
        )
        return

    def forward(self, timesteps):
        return self.time_embed(self.CTS(timesteps))
    


if __name__ == "__main__":
    CTS = ContinuousTimesteps(num_channels=6, flip_sin_to_cos=False, downscale_freq_shift=1, scale=1)
    # t_emb = CTS(timesteps=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    t_emb = CTS(timesteps=torch.tensor([1, 2, 3, 4, 5, 6]))
    print(t_emb)
    print()

