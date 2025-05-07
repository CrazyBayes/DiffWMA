from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
import torch.nn.functional as F
from modules.sampling.functions import default_sample_fn, policy_guided_sample_fn
from torch.optim import Adam

from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple(
    "Sample", "trajectories chains recons_after_guide recons_before_guide"
)


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        args,
        model,
        n_timesteps=20,
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
        noise_sched_tau=1.0,
        action_condition_noise_scale=1.0,
    ):
        super().__init__()
        self.args = args
        self.observation_dim = args.obs_shape
        self.action_dim = args.n_actions
        self.transition_dim = args.state_shape # obs + reward + terminals
        self.model = model
        self.action_condition_noise_scale = action_condition_noise_scale

        betas = cosine_beta_schedule(n_timesteps, tau=noise_sched_tau)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        ####需要定义优化器
        self.params = list(self.model.parameters())
        self.optimizer = Adam(self.model.parameters(), lr=args.DWN_lr)

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )

        coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", coef1)
        self.register_buffer("posterior_mean_coef2", coef2)

        ## initialize objective
        self.loss_fn = Losses[loss_type]()

    @property
    def device(self):
        return self.betas.device

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    def p_sample(self, x_condition, x, t: int):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x_condition = x_condition, x = x, t = batched_times)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    def p_mean_variance(self, x_condition, x, t):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = self.model(x_condition, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=prediction)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def p_sample_loop(
        self,
        shape,
        cond,
        normalizer=None,
        return_sequence=False,
        verbose=True,
        return_chain=False,
    ):
        img = torch.randn(shape, device=self.betas.device)
        x = apply_conditioning(img, cond)
        seq = [img]
        for t in reversed(range(0, self.n_timesteps)):
            #self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(x, img, t)
            x = apply_conditioning(img, cond)
            if return_sequence: ###这个地方把condition也包括进去了
                seq.append(img)
        return img #最后一个才是我们要的结果

    def conditional_sample(
        self,
        cond,
        normalizer=None
    ):
        """
        conditions : [ (time, state), ... ]
        """
        #batch_size = len(cond[0])
        #print(cond.shape)
        first_dim, second_dim, _ = cond.shape
        #shape = (self.args.batch_size_run, batch_size, self.args.state_shape)
        shape = (first_dim, second_dim, self.args.state_shape+1+self.args.obs_shape)

        return self.p_sample_loop(
            shape, cond, normalizer=normalizer
        )

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        traj_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        #traj_noisy = traj_noisy.unsqueeze(2).expand(-1,-1, self.args.n_agents, -1)
        traj_noisy = apply_conditioning(traj_noisy, cond)


        traj_recon = self.model(traj_noisy, t)
        #traj_recon = apply_conditioning(traj_recon, cond)

        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        #target = target.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)

        loss = self.loss_fn(traj_recon, target)
        #loss_metrics = {}
        '''
        with torch.no_grad():
            loss_metrics["obs_mse_loss"] = F.mse_loss(
                traj_recon[:, :, : self.observation_dim],
                target[:, :, : self.observation_dim],
            ).item()
            loss_metrics["reward_mse_loss"] = F.mse_loss(
                traj_recon[:, :, -2], target[:, :, -2]
            ).item()
            loss_metrics["term_mse_loss"] = F.mse_loss(
                traj_recon[:, :, -1], target[:, :, -1]
            ).item()
            '''
        return loss, traj_recon#, loss_metrics

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def train_DWN(self, x, cond, DWM_W): #训练Denoiser
        loss, cons_tra  = self.loss(x, cond)
        DWM_W = DWM_W.unsqueeze(dim=-1)
        loss = loss * DWM_W
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.args.DWN_grad_norm_clip)
        self.optimizer.step()

        return loss, cons_tra

    def forward(self, cond):
        return self.conditional_sample(cond)
