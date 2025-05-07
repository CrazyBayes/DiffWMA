import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from .mlp_denoiser_network import ResidualMLP

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(
            self,
            dim,
            is_random: bool = False,
    ):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args =args
        d_in = args.state_shape + 1 + args.obs_shape
        d_out = d_in
        cond_dim = args.obs_shape + args.n_actions +  args.rnn_hidden_dim + args.n_agents
        self.embed_dim = args.DWN_proj
        self.cond_dim = cond_dim
        self.d_in = d_in

        self.proj = nn.Linear(d_in + cond_dim, self.embed_dim)



        self.MLP_denoiser = ResidualMLP(
            input_dim=self.embed_dim,
            width=args.DWM_hidden_dim,
            depth=args.DWM_num_layers,
            output_dim= d_out,
            activation=args.DWM_activation,
            layer_norm=args.DWM_layer_norm
        )
        # time embeddings
        learned_sinusoidal_cond = False,
        random_fourier_features = True,
        learned_sinusoidal_dim = 16
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(args.DWM_embed_dim)
            fourier_dim = args.DWM_embed_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, args.DWM_embed_dim),
            nn.SiLU(),
            nn.Linear(args.DWM_embed_dim, self.args.DWN_proj)
        )


    def forward(
                self,
                traj: torch.Tensor,
                #condition: torch.Tensor,
                timesteps: torch.Tensor
        ) -> torch.Tensor:
            # traj[:, 1:, :-1] = traj[:, 1:, :-1] * self.scale_obs # scale obs
            # traj[:, :, -1:] = traj[:, :, -1:] * self.scale_obs # scale rew
            #x = torch.cat([traj, condition], dim=-1)  ##传过来的
            x = traj
            #b, h, d = traj.shape
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.unsqueeze(1).expand(-1, self.args.n_agents, -1)
            if x.ndimension() > time_embed.ndimension():
                time_embed = time_embed.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
            x = self.proj(x) + time_embed
            y = self.MLP_denoiser(x)

            #y = y.reshape(b, h, d)
            return y