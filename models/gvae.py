from typing import Optional
from itertools import chain
from functools import partial
import pdb
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
# import dgl
# from dgl.ops import edge_softmax
# import dgl.function as fn
# from dgl.utils import expand_as_pair
import pdb

from math import exp

from models.gnn import *

class PIDControl():
    """docstring for ClassName"""
    def __init__(self):
        """define them out of loop"""
        # self.exp_KL = exp_KL
        self.I_k1 = 0.0
        self.W_k1 = 1.0
        self.e_k1 = 0.0
        
    def _Kp_fun(self, Err, scale=1):
        return 1.0/(1.0 + float(scale)*exp(Err))
        
    def pid(self, exp_KL, kl_divergence, Kp=0.02, Ki=-0.001, Kd=0.01):
        """
        position PID algorithm
        Input: KL_loss
        return: weight for KL loss, beta
        """
        error_k = exp_KL - kl_divergence
        ## comput U as the control factor
        Pk = Kp * self._Kp_fun(error_k)+0.5 #here
        Ik = self.I_k1 + Ki * error_k
        
        ## window up for integrator
        if self.W_k1 < 1:
            Ik = self.I_k1
            
        Wk = Pk + Ik
        self.W_k1 = Wk
        self.I_k1 = Ik
        
        ## min and max value
        if Wk < 1e-6:
            Wk = 1e-6
        
        return Wk, error_k

class GraphVAE(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            use_mask: bool = False,
            mask_rate: float = 0.3,
            encoder_type: str = "gcn",
            decoder_type: str = "mlp",
            loss_fn: str = "bce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            # concat_hidden: bool = False,
            vae: bool = True,
         ):
        super(GraphVAE, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = False
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._use_mask = use_mask
        self._vae = vae
        # self.loss_weight = 1
        # self.kl_weight = 1
        self.exp_kl = 0
        self.exp_kl_step = 0.01
        self.exp_kl_max = 1
    
        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        if self._vae:
            self.encoder = setup_module(
                m_type=encoder_type,
                enc_dec="encoding",
                in_dim=in_dim,
                num_hidden=enc_num_hidden,
                out_dim=2 * enc_num_hidden,
                num_layers=num_layers,
                nhead=enc_nhead,
                nhead_out=enc_nhead,
                concat_out=True,
                activation=activation,
                dropout=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm,
            )
        else:
            self.encoder = setup_module(
                m_type=encoder_type,
                enc_dec="encoding",
                in_dim=in_dim,
                num_hidden=enc_num_hidden,
                out_dim=enc_num_hidden,
                num_layers=num_layers,
                nhead=enc_nhead,
                nhead_out=enc_nhead,
                concat_out=True,
                activation=activation,
                dropout=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm,
            )

        # build decoder for attribute prediction
        if self._decoder_type in ("mlp", "linear"):
            self.decoder = nn.Sequential(
            MLP(input_dim=dec_in_dim, 
                           output_dim=in_dim, 
                           num_layers=2, 
                           hidden_dim=dec_num_hidden, 
                           activation=activation),
            nn.Sigmoid())
        else:
            
            self.decoder = nn.Sequential(
            setup_module(
                m_type=decoder_type,
                enc_dec="decoding",
                in_dim=dec_in_dim,
                num_hidden=dec_num_hidden,
                out_dim=in_dim,
                num_layers=1,
                nhead=nhead,
                nhead_out=nhead_out,
                activation=activation,
                dropout=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm,
                concat_out=True,
            ),
            nn.Sigmoid())
        

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if self._concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.pid = PIDControl()
        
        self.z_all = []

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss(reduction='sum')
        # elif loss_fn == "sce":
        #     criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_fn == "bce":
            criterion = nn.BCELoss(reduction='sum')
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, device=std.device)
        return mu + eps*std
    
    def update_exp_kl(self):
        assert self.training, "exp_kl should only be updated during training"
        self.exp_kl = self.exp_kl + self.exp_kl_step
        if self.exp_kl > self.exp_kl_max:
            self.exp_kl = self.exp_kl_max
        

    def forward(self, g, x):
        # ---- attribute reconstruction ----
        if self._use_mask:
            # loss, kl_loss = self.mask_attr_prediction(g, x)
            raise NotImplementedError
        else:
            loss, kl_loss = self.mask_attr_prediction_no_mask(g, x)
        if self._vae:
            loss_item = {"loss": loss.item(), "kl_loss": kl_loss.item()}
            # loss = self.loss_weight*loss + self.kl_weight*kl_loss
            self.kl_weight, _ = self.pid.pid(self.exp_kl, kl_loss.item())
            loss = self.kl_weight*kl_loss + loss
        else:
            loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    # def mask_attr_prediction(self, g, x):
    #     pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

    #     if self._drop_edge_rate > 0:
    #         use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
    #     else:
    #         use_g = pre_use_g

    #     enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
    #     if self._vae:
    #         enc_mu = enc_rep[:, : self._output_hidden_size]
    #         enc_logvar = enc_rep[:, self._output_hidden_size:]
    #         enc_rep = self.reparameterize(enc_mu, enc_logvar)

    #     if self._concat_hidden:
    #         enc_rep = torch.cat(all_hidden, dim=1)

    #     # ---- attribute reconstruction ----
    #     rep = self.encoder_to_decoder(enc_rep)

    #     if self._decoder_type not in ("mlp", "linear"):
    #         # * remask, re-mask
    #         rep[mask_nodes] = 0

    #     if self._decoder_type in ("mlp", "linear") :
    #         recon = self.decoder(rep)
    #     else:
    #         recon = self.decoder(pre_use_g, rep)

    #     x_init = x[mask_nodes]
    #     x_rec = recon[mask_nodes]

    #     loss = self.criterion(x_rec, x_init) # recon los
    #     kl_loss = 0
    #     if self._vae:
    #         kl_loss = -0.5 * torch.sum(1 + enc_logvar - enc_mu.pow(2) - enc_logvar.exp())

    #     return loss, kl_loss

    def mask_attr_prediction_no_mask(self, g, x):
        use_g = g.clone()
        use_x = x.clone()

        if self._drop_edge_rate > 0:
            use_g, _ = drop_edge(use_g, self._drop_edge_rate, return_edges=True)

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._vae:
            enc_mu = enc_rep[:, : self._output_hidden_size]
            enc_logvar = enc_rep[:, self._output_hidden_size:]
            enc_rep = self.reparameterize(enc_mu, enc_logvar)
            
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        # ---- memorize the latent representation during training----
        if self.training:
            self.z_all.append(enc_rep.detach().unsqueeze(0))
        
        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)

        x_init = x
        x_rec = recon

        loss = self.criterion(x_rec, x_init)/x.shape[0]
        kl_loss = 0
        if self._vae:
            kl_loss = -0.5 * torch.sum(1 + enc_logvar - enc_mu.pow(2) - enc_logvar.exp())/x.shape[0] 
            
        return loss, kl_loss

    def cal_mean_z(self):
        assert self.training
        self.z_mean = torch.mean(torch.cat(self.z_all, dim=0), dim=0)
        self.z_all = []
        return self.z_mean
    
    def embed(self, g, x):
        rep = self.encoder(g, x)
        if self._vae:
            rep_mu = rep[..., :self._output_hidden_size]
            rep_logvar = rep[..., self._output_hidden_size:]
            rep = self.reparameterize(rep_mu, rep_logvar)
            
        return rep, rep_mu, rep_logvar

    def decode(self, g, enc_rep):
        rep = self.encoder_to_decoder(enc_rep)
        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(g, rep)
        return recon
    
    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
