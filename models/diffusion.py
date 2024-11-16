import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from models.gnn import setup_module

import matplotlib.pyplot as plt
import numpy as np

from utils import data_loader
from models.positional_embeddings import PositionalEmbedding
from models import denoiser
# from models.guide import rewardGuide, classifierGuide, toyGuide
# diffusion_model.py
from tqdm import tqdm
prev_fs = 0
prev_rs = 0
# class Block(nn.Module):
#     def __init__(self, size: int):
#         super().__init__()

#         self.ff = nn.Linear(size, size)
#         self.act = nn.GELU()

#     def forward(self, x: torch.Tensor):
#         return x + self.act(self.ff(x))


# class MLP(nn.Module):
#     def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
#                  time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
#         super().__init__()

#         self.time_mlp = PositionalEmbedding(emb_size, time_emb)
#         self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
#         self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

#         concat_size = len(self.time_mlp.layer) + \
#             len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
#         layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
#         for _ in range(hidden_layers):
#             layers.append(Block(hidden_size))
#         layers.append(nn.Linear(hidden_size, 2))
#         self.joint_mlp = nn.Sequential(*layers)

#     def forward(self, x, t):
#         x1_emb = self.input_mlp1(x[:, 0])
#         x2_emb = self.input_mlp2(x[:, 1])
#         t_emb = self.time_mlp(t)
#         x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
#         x = self.joint_mlp(x)
#         return x


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear",
                 pred_x0 = False,
                 device="cuda"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32, device=device ) ** 2
        elif beta_schedule == "cosine":
            self.betas = beta_end + 0.5 * \
                (beta_start - beta_end) * \
                (1 + torch.cos(torch.linspace(0, np.pi, num_timesteps,device=device)))

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.pred_x0 = pred_x0
        
    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        if self.pred_x0:
            pred_original_sample = model_output
        else:
            pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output, device=model_output.device)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def add_onestep_noise(self, x_t, x_n, t):
        s1 = torch.sqrt(1-self.betas[t])
        s2 = self.betas[t]
        
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        
        return s1 * x_t + s2 * x_n

    def __len__(self):
        return self.num_timesteps


class NoiseScheduler_yTmean():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear",
                 pred_x0 = False,
                 device="cuda"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32, device=device ) ** 2
        elif beta_schedule == "cosine":
            self.betas = beta_end + 0.5 * \
                (beta_start - beta_end) * \
                (1 + torch.cos(torch.linspace(0, np.pi, num_timesteps,device=device)))

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef3 = 1 + (self.sqrt_alphas_cumprod - 1) * (torch.sqrt(self.alphas) + torch.sqrt(self.alphas_cumprod_prev)) / (
                                    (1. - self.alphas_cumprod))
        # assert pred_x0==False
        self.pred_x0 = pred_x0
        
    def reconstruct_x0(self, x_t, t, noise, y_T_mean):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise - (s1-1) * y_T_mean

    def q_posterior(self, x_0, x_t, t, y_T_mean):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s3 = self.posterior_mean_coef3[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        s3 = s3.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t + s3 * y_T_mean
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample, y_T_mean):
        t = timestep
        if self.pred_x0:
            pred_original_sample = model_output
        else:
            pred_original_sample = self.reconstruct_x0(sample, t, model_output, y_T_mean)

        variance = 0
        if t > 0:
            pred_prev_sample = self.q_posterior(pred_original_sample, sample, t, y_T_mean)
            
            noise = torch.randn_like(model_output, device=model_output.device)
            variance = (self.get_variance(t) ** 0.5) * noise
        else:
            pred_prev_sample = pred_original_sample

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    # def add_noise(self, x_start, x_noise, timesteps):
    #     s1 = self.sqrt_alphas_cumprod[timesteps]
    #     s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

    #     s1 = s1.reshape(-1, 1)
    #     s2 = s2.reshape(-1, 1)

    #     return s1 * x_start + s2 * x_noise

    def add_noise(self, x_start, x_0_hat, x_noise, timesteps):
        """
        x_0_hat: prediction of pre-trained guidance classifier; can be extended to represent 
            any prior mean setting at timestep T.
        """
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        # q(y_t | y_0, x)
        y_t = s1 * x_start + (1 - s1) * x_0_hat + s2 * x_noise
        return y_t

    def add_onestep_noise(self, x_t, x_n, t, y_T_mean):
        s1 = torch.sqrt(1-self.betas[t])
        s2 = self.betas[t]
        
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        
        return s1 * x_t + s2 * x_n + (1 - s1) * y_T_mean

    def __len__(self):
        return self.num_timesteps


class DiffusionModel(nn.Module):
    def __init__(self, gnn_type, 
                in_dim, 
                noise_emb_dim, 
                hidden_dim, 
                num_layers, 
                activation, 
                feat_drop, 
                attn_drop, 
                negative_slope, 
                residual, 
                norm, 
                enc_nhead,
                mlp_layers,
                num_timesteps=100,
                beta_schedule="linear",
                guidance_scale = 10,
                pred_x0 = False,
                device="cuda"):
        super().__init__()
        
        self.gnn_type = gnn_type
        self.in_dim = in_dim
        self.noise_emb_dim = noise_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        self.norm = norm
        self.enc_nhead = enc_nhead
        self.mlp_layers = mlp_layers
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        
        # self.model = denoiser.DenoiserMLP(gnn_type, 
                                    # in_dim, 
                                    # noise_emb_dim, 
                                    # hidden_dim, 
                                    # num_layers, 
                                    # activation, 
                                    # feat_drop, 
                                    # attn_drop, 
                                    # negative_slope, 
                                    # residual, 
                                    # norm, 
                                    # enc_nhead,
                                    # mlp_layers,)
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            pred_x0 = pred_x0,
            device=device)
        
        self.model = denoiser.DenoiserUnet(gnn_type, 
                                in_dim, 
                                noise_emb_dim, 
                                hidden_dim, 
                                num_layers, 
                                activation, 
                                feat_drop, 
                                attn_drop, 
                                negative_slope, 
                                residual, 
                                norm, 
                                enc_nhead,
                                mlp_layers,)
    
        
        # self.guide = rewardGuide(in_dim=in_dim,)
        self.guide = classifierGuide(in_dim=in_dim,)
        # self.guide = toyGuide(in_dim=in_dim,)
        self.guidance_scale = guidance_scale
        self.resample_steps = 5
        
        
    def train_step(self, batch, g, y):
        assert batch.shape[0]==1
        assert len(batch.shape)==3
        noise = torch.randn(batch.shape, device=batch.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_timesteps, (batch.shape[0], ), device=batch.device
        ).long()

        noisy = self.noise_scheduler.add_noise(batch, noise, timesteps)
        pred = self.model(noisy, timesteps + 1, g)
        # if self.noise_scheduler.pred_x0:
            # pred = torch.sigmoid(pred)-0.5
            
        if self.noise_scheduler.pred_x0:
            loss = F.mse_loss(pred, batch)
        else:
            loss = F.mse_loss(pred, noise)
        
        batch_ = batch.clone().detach()
        # if self.noise_scheduler.pred_x0:
        #     batch_ = pred.clone().detach()
        # else:
        #     batch_ = self.noise_scheduler.reconstruct_x0(batch.clone().detach(), timesteps, pred.clone().detach())
        
        y_pred = self.guide(batch_, g, timesteps)
        loss_guide = F.mse_loss(F.sigmoid(y_pred), y)

        # loss_guide = torch.zeros(1)

        return loss, loss_guide
    
    def sample(self, g, cond):
        assert cond.shape[0]==1
        batch_num = cond.shape[0]
        node_num = cond.shape[1]
        sample = torch.randn(list(cond.shape[:-1])+[self.in_dim], device=cond.device)
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        
        fs = []
        rs = []
        for i, t in enumerate(timesteps):
            ts = t
            fs.append(sample[0,0,0].item())
            
            ts = torch.from_numpy(np.repeat(ts, batch_num)).long().to(cond.device)
            with torch.no_grad():
                residual = self.model(sample, ts + 1, g)
                # if self.noise_scheduler.pred_x0: 
                    # residual = torch.sigmoid(residual)-0.5
                rs.append(residual[0,0,0].item())
            sample = self.noise_scheduler.step(residual, ts[0], sample)
        #plot fs and save: plt.plot(
        global prev_fs 
        global prev_rs
        plt.figure(1)
        plt.plot(fs)
        plt.plot(prev_fs)
        plt.savefig("1.png")
        plt.clf()
        plt.figure(2)
        plt.plot(rs)
        plt.plot(prev_rs)
        plt.savefig("2.png")
        plt.clf()
        
        prev_fs = fs
        prev_rs = rs
        return sample
    
    def sample_with_cond_guidance(self, g, cond, gvae):
        assert cond.shape[0]==1
        batch_num = cond.shape[0]
        node_num = cond.shape[1]
        sample = torch.randn(list(cond.shape[:-1])+[self.in_dim], device=cond.device)
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        # self.guide.get_predprob(cond.squeeze(0),g)
        for i, t in enumerate(timesteps):
            t = torch.from_numpy(np.repeat(t, batch_num)).long().to(cond.device)
            with torch.no_grad():
                residual = self.model(sample, t+1, g)
                # if self.noise_scheduler.pred_x0:
                #     residual = torch.sigmoid(residual)-0.5
            sample_next = self.noise_scheduler.step(residual, t[0], sample)
            with torch.enable_grad():
                assert self.guide is not None
                variance = self.noise_scheduler.get_variance(t)
                grad = self.guide.gradients(sample, cond, g, variance, t, self.noise_scheduler.reconstruct_x0, residual, gvae)
                grad = grad * self.guidance_scale
            if t>=0:
                sample = sample_next + grad
            elif t==0:
                sample = sample_next
                
        return sample
    
    def sample_with_cond_mask(self, g, cond, mask):
        assert cond.shape[0]==1 and mask.shape[0]==1
        mask = mask.unsqueeze(-1)
        batch_num = cond.shape[0]
        node_num = cond.shape[1]
        sample = torch.randn(list(cond.shape[:-1])+[self.in_dim], device=cond.device)
        assert sample.shape == cond.shape
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        for i, t in enumerate(timesteps):
            ts = torch.from_numpy(np.repeat(t, batch_num)).long().to(cond.device) 
            for u in range(self.resample_steps):
                with torch.no_grad():
                    noise = torch.randn(sample.shape, device=sample.device)
                    noise2 = torch.randn(sample.shape, device=sample.device)
                    if t>0:
                        known = self.noise_scheduler.add_noise(cond, noise, t-1)
                    else:
                        known = cond
                    residual = self.model(sample, ts+ 1, g)
                    # if self.noise_scheduler.pred_x0:
                    #     residual = torch.sigmoid(residual)-0.5
                    sample_next = self.noise_scheduler.step(residual, t, sample)
                    sample_next = sample_next * mask + known * (1-mask)
                    if (u<self.resample_steps-1) and (t>0):
                        sample = self.noise_scheduler.add_onestep_noise(sample_next, noise2, t-1)
            sample = sample_next
            
        return sample
            
class AdvicedDiffusionModel(nn.Module):
    def __init__(self, gnn_type, 
                in_dim, 
                noise_emb_dim, 
                hidden_dim, 
                num_layers, 
                activation, 
                feat_drop, 
                attn_drop, 
                negative_slope, 
                residual, 
                norm, 
                enc_nhead,
                mlp_layers,
                num_timesteps=500,
                beta_schedule="linear",
                guidance_scale = 10,
                pred_x0 = True,
                device="cuda",
                num_advisors=1):
        super().__init__()
        
        self.gnn_type = gnn_type
        self.in_dim = in_dim
        self.noise_emb_dim = noise_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        self.norm = norm
        self.enc_nhead = enc_nhead
        self.mlp_layers = mlp_layers
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        
        # self.model = denoiser.DenoiserMLP(gnn_type, 
                                    # in_dim, 
                                    # noise_emb_dim, 
                                    # hidden_dim, 
                                    # num_layers, 
                                    # activation, 
                                    # feat_drop, 
                                    # attn_drop, 
                                    # negative_slope, 
                                    # residual, 
                                    # norm, 
                                    # enc_nhead,
                                    # mlp_layers,)
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            pred_x0 = pred_x0,
            device=device)
        
        self.model = denoiser.DenoiseAdvisor(gnn_type, 
                                            in_dim, 
                                            noise_emb_dim, 
                                            hidden_dim, 
                                            num_advisors,
                                            num_layers, 
                                            activation, 
                                            feat_drop, 
                                            attn_drop, 
                                            negative_slope, 
                                            residual, 
                                            norm, 
                                            enc_nhead,
                                            mlp_layers)
        
        
        # self.guide = rewardGuide(in_dim=in_dim,)
        # self.guide = classifierGuide(in_dim=in_dim,)
        # self.guide = toyGuide(in_dim=in_dim,)
        # self.guidance_scale = guidance_scale
        # self.resample_steps = 5
        
        
    def train_step(self, batch, g, y, advisors=None, y_T_mean=None):
        assert batch.shape[0]==1
        assert len(batch.shape)==3
        noise = torch.randn(batch.shape, device=batch.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_timesteps, (batch.shape[0], ), device=batch.device
        ).long()

        noisy = self.noise_scheduler.add_noise(batch, noise, timesteps)
        cond = self.model.conditioning(y, g)
        pred = self.model(noisy, timesteps, g, cond, advisors)
        # if self.noise_scheduler.pred_x0:
            # pred = torch.sigmoid(pred)-0.5
            
        if self.noise_scheduler.pred_x0:
            loss = F.mse_loss(pred, batch)
        else:
            loss = F.mse_loss(pred, noise)
        
        # batch_ = batch.clone().detach()
        # if self.noise_scheduler.pred_x0:
        #     batch_ = pred.clone().detach()
        # else:
        #     batch_ = self.noise_scheduler.reconstruct_x0(batch.clone().detach(), timesteps, pred.clone().detach())
        
        # y_pred = self.guide(batch_, g, timesteps)
        # loss_guide = F.mse_loss(F.sigmoid(y_pred), y)

        # loss_guide = torch.zeros(1)

        return loss, None
    
    def train_conditioner(self, y, g, lpsi_y):
        y_pred = self.model.conditional(y, g)
        
        loss = F.mse_loss(y_pred, lpsi_y)
        
        return loss
    
    def lpsi(self, y, g):
        g = self.model.conditional.draw_adj(g)
        norm = self.model.conditional.normalize_adj(g)
        seed, _, coverage = self.model.conditional.LPSI_coverage(g, norm, y, 0.4)

        return seed, coverage.unsqueeze(-1)
    
    def sample(self, g, cond, advisors=None, y_T_mean=None):
        assert cond.shape[0]==1
        batch_num = cond.shape[0]
        node_num = cond.shape[1]
        sample = torch.randn(list(cond.shape[:-1])+[self.in_dim], device=cond.device)
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        
        # fs = []
        # rs = []
        cond = cond.squeeze(0)
        cond = self.model.conditioning(cond, g)
        for i, t in enumerate(timesteps):
            ts = t
            
            ts = torch.from_numpy(np.repeat(ts, batch_num)).long().to(cond.device)
            with torch.no_grad():
                residual = self.model(sample, ts, g, cond, advisors)
                # if self.noise_scheduler.pred_x0: 
                    # residual = torch.sigmoid(residual)-0.5
                # rs.append(residual[0,0,0].item())
                # if t<5:
                #     print('1')
                sample = self.noise_scheduler.step(residual, t, sample)
                # sample = residual  # 陷入不动点
                # fs.append(sample[0,0,0].item())
                
        #plot fs and save: plt.plot(
        # global prev_fs 
        # global prev_rs
        # plt.figure(1)
        # plt.plot(fs)
        # plt.plot(prev_fs)
        # plt.legend(["current", "previous"])
        # plt.savefig("1_.png")
        # plt.clf()
        # plt.figure(2)
        # plt.plot(rs)
        # plt.plot(prev_rs)
        # plt.legend(["current", "previous"])
        # plt.savefig("2_.png")
        # plt.clf()
        
        # prev_fs = fs
        # prev_rs = rs
        return sample

class AdvicedDiffusionModel_yTmean(nn.Module):
    def __init__(self, gnn_type, 
                in_dim, 
                noise_emb_dim, 
                hidden_dim, 
                num_layers, 
                activation, 
                feat_drop, 
                attn_drop, 
                negative_slope, 
                residual, 
                norm, 
                enc_nhead,
                mlp_layers,
                num_timesteps=500,
                beta_schedule="linear",
                guidance_scale = 10,
                pred_x0 = True,
                device="cuda",
                num_advisors=1):
        super().__init__()
        
        self.gnn_type = gnn_type
        self.in_dim = in_dim
        self.noise_emb_dim = noise_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        self.norm = norm
        self.enc_nhead = enc_nhead
        self.mlp_layers = mlp_layers
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        
        # self.model = denoiser.DenoiserMLP(gnn_type, 
                                    # in_dim, 
                                    # noise_emb_dim, 
                                    # hidden_dim, 
                                    # num_layers, 
                                    # activation, 
                                    # feat_drop, 
                                    # attn_drop, 
                                    # negative_slope, 
                                    # residual, 
                                    # norm, 
                                    # enc_nhead,
                                    # mlp_layers,)
        self.noise_scheduler = NoiseScheduler_yTmean(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            pred_x0 = pred_x0,
            device=device)
        
        self.model = denoiser.DenoiseAdvisor(gnn_type, 
                                            in_dim, 
                                            noise_emb_dim, 
                                            hidden_dim, 
                                            num_advisors,
                                            num_layers, 
                                            activation, 
                                            feat_drop, 
                                            attn_drop, 
                                            negative_slope, 
                                            residual, 
                                            norm, 
                                            enc_nhead,
                                            mlp_layers)
        
        
        # self.guide = rewardGuide(in_dim=in_dim,)
        # self.guide = classifierGuide(in_dim=in_dim,)
        # self.guide = toyGuide(in_dim=in_dim,)
        # self.guidance_scale = guidance_scale
        # self.resample_steps = 5
        
        
    def train_step(self, batch, g, y, advisors=None, y_T_mean=None):
        assert batch.shape[0]==1
        assert len(batch.shape)==3
        noise = torch.randn(batch.shape, device=batch.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_timesteps, (batch.shape[0], ), device=batch.device
        ).long()

        noisy = self.noise_scheduler.add_noise(batch, y_T_mean, noise, timesteps)
        cond = self.model.conditioning(y, g)
        pred = self.model(noisy, timesteps, g, cond, advisors)
        # if self.noise_scheduler.pred_x0:
            # pred = torch.sigmoid(pred)-0.5
            
        if self.noise_scheduler.pred_x0:
            loss = F.mse_loss(pred, batch)
        else:
            loss = F.mse_loss(pred, noise)
        
        # batch_ = batch.clone().detach()
        # if self.noise_scheduler.pred_x0:
        #     batch_ = pred.clone().detach()
        # else:
        #     batch_ = self.noise_scheduler.reconstruct_x0(batch.clone().detach(), timesteps, pred.clone().detach())
        
        # y_pred = self.guide(batch_, g, timesteps)
        # loss_guide = F.mse_loss(F.sigmoid(y_pred), y)

        # loss_guide = torch.zeros(1)

        return loss, None
    
    def train_conditioner(self, y, g, lpsi_y):
        y_pred = self.model.conditional(y, g)
        
        loss = F.mse_loss(y_pred, lpsi_y)
        
        return loss
    
    def lpsi(self, y, g):
        g = self.model.conditional.draw_adj(g)
        norm = self.model.conditional.normalize_adj(g)
        seed, _, coverage = self.model.conditional.LPSI_coverage(g, norm, y, 0.4)

        return seed, coverage.unsqueeze(-1)
    
    def sample(self, g, cond, advisors=None, y_T_mean=None):
        assert cond.shape[0]==1
        batch_num = cond.shape[0]
        node_num = cond.shape[1]
        sample = torch.randn(list(cond.shape[:-1])+[self.in_dim], device=cond.device)
        if y_T_mean is not None:
            sample = sample + y_T_mean
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        
        # fs = []
        # rs = []
        cond = cond.squeeze(0)
        cond = self.model.conditioning(cond, g)
        for i, t in enumerate(timesteps):
            ts = t
            
            ts = torch.from_numpy(np.repeat(ts, batch_num)).long().to(cond.device)
            with torch.no_grad():
                residual = self.model(sample, ts, g, cond, advisors)
                # if self.noise_scheduler.pred_x0: 
                #     residual = torch.sigmoid(residual)-0.5
                # rs.append(residual[0,0,0].item())
                # if t<5:
                #     print('1')
                sample = self.noise_scheduler.step(residual, t, sample, y_T_mean)
                # sample = residual  # 陷入不动点
                # fs.append(sample[0,0,0].item())
                
        # #plot fs and save: plt.plot(
        # global prev_fs 
        # global prev_rs
        # plt.figure(1)
        # plt.plot(fs)
        # plt.plot(prev_fs)
        # plt.legend(["current", "previous"])
        # plt.savefig("1_.png")
        # plt.clf()
        # plt.figure(2)
        # plt.plot(rs)
        # plt.plot(prev_rs)
        # plt.legend(["current", "previous"])
        # plt.savefig("2_.png")
        # plt.clf()
        
        # prev_fs = fs
        # prev_rs = rs
        
        # norm
        sample_min = sample.min()
        sample_max = sample.max()
        sample = (sample-sample_min)/(sample_max-sample_min)
        return sample


# def train_model(experiment_name="base",
#                 dataset="dino",
#                 train_batch_size=32,
#                 eval_batch_size=1000,
#                 num_epochs=200,
#                 learning_rate=1e-3,
#                 num_timesteps=50,
#                 beta_schedule="linear",
#                 embedding_size=128,
#                 hidden_size=128,
#                 hidden_layers=3,
#                 time_embedding="sinusoidal",
#                 input_embedding="sinusoidal",
#                 save_images_step=1):

#     dataset = data_loader.get_dataset(dataset)
#     dataloader = DataLoader(
#         dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

#     model = denoiser.Denoiser(gnn_type, 
#                                 in_dim, 
#                                 noise_emb_dim, 
#                                 hidden_dim, 
#                                 num_layers, 
#                                 activation, 
#                                 feat_drop, 
#                                 attn_drop, 
#                                 negative_slope, 
#                                 residual, 
#                                 norm, 
#                                 enc_nhead,
#                                 mlp_layers)

#     noise_scheduler = NoiseScheduler(
#         num_timesteps=num_timesteps,
#         beta_schedule=beta_schedule)

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=learning_rate,
#     )

#     global_step = 0
#     frames = []
#     losses = []
#     print("Training model...")
#     for epoch in range(num_epochs):
#         model.train()
#         progress_bar = tqdm(total=len(dataloader))
#         progress_bar.set_description(f"Epoch {epoch}")
#         for step, batch in enumerate(dataloader):
#             batch = batch[0]
#             noise = torch.randn(batch.shape)
#             timesteps = torch.randint(
#                 0, noise_scheduler.num_timesteps, (batch.shape[0],)
#             ).long()

#             noisy = noise_scheduler.add_noise(batch, noise, timesteps)
#             noise_pred = model(noisy, timesteps)
#             loss = F.mse_loss(noise_pred, noise)
#             loss.backward(loss)

#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             optimizer.zero_grad()

#             progress_bar.update(1)
#             logs = {"loss": loss.detach().item(), "step": global_step}
#             losses.append(loss.detach().item())
#             progress_bar.set_postfix(**logs)
#             global_step += 1
#         progress_bar.close()

#         if epoch % save_images_step == 0 or epoch == num_epochs - 1:
#             # generate data with the model to later visualize the learning process
#             model.eval()
#             sample = torch.randn(eval_batch_size, 2)
#             timesteps = list(range(len(noise_scheduler)))[::-1]
#             for i, t in enumerate(tqdm(timesteps)):
#                 t = torch.from_numpy(np.repeat(t, eval_batch_size)).long()
#                 with torch.no_grad():
#                     residual = model(sample, t)
#                 sample = noise_scheduler.step(residual, t[0], sample)
#             frames.append(sample.numpy())

#     print("Saving model...")
#     outdir = f"exps/{experiment_name}"
#     os.makedirs(outdir, exist_ok=True)
#     torch.save(model.state_dict(), f"{outdir}/model.pth")

#     print("Saving images...")
#     imgdir = f"{outdir}/images"
#     os.makedirs(imgdir, exist_ok=True)
#     frames = np.stack(frames)
#     xmin, xmax = -6, 6
#     ymin, ymax = -6, 6
#     for i, frame in enumerate(frames):
#         plt.figure(figsize=(10, 10))
#         plt.scatter(frame[:, 0], frame[:, 1])
#         plt.xlim(xmin, xmax)
#         plt.ylim(ymin, ymax)
#         plt.savefig(f"{imgdir}/{i:04}.png")
#         plt.close()

#     print("Saving loss as numpy array...")
#     np.save(f"{outdir}/loss.npy", np.array(losses))

#     print("Saving frames...")
#     np.save(f"{outdir}/frames.npy", frames)

# if __name__ == "__main__":
#     train_model()
