import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model.base_model import Base_Model
from utils.pytorch_helper import FF, get_init_function

class corrSH(Base_Model):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def define_parameters(self):
        self.enc = Encoder(self.data.vocab_size, self.hparams.dim_hidden,
                              self.hparams.dim_hidden,
                              self.hparams.num_layers)
        self.h_to_mu = nn.Linear(self.hparams.dim_hidden, self.hparams.num_features)
        self.h_to_std = nn.Linear(self.hparams.dim_hidden, self.hparams.num_features)
        self.h_to_U = nn.ModuleList([nn.Linear(self.hparams.dim_hidden, self.hparams.num_features) for i in range(self.hparams.rank)])
        self.dec = Decoder(self.hparams.num_features, self.data.vocab_size)

    def encode(self, X):
        h = self.enc(X.sign())
        mu = self.h_to_mu(h)
        std = F.softplus(self.h_to_std(h))
        rs = []
        for i in range(self.hparams.rank):
            rs.append((1 / self.hparams.rank) * torch.tanh(self.h_to_U[i](h)))
        u_pert = tuple(rs)
        u_perturbation = torch.stack(u_pert, 2)

        return mu, std, u_perturbation
    
    def reparameterize(self, mu, std, u_pert):
        eps = torch.randn_like(mu)
        eps_corr = torch.randn((u_pert.shape[0], u_pert.shape[2], 1)).to(self.hparams.device)
        # Reparameterisation trick for low-rank-plus-diagonal Gaussian
        z = mu + eps * std + torch.matmul(u_pert, eps_corr).squeeze()
        
        prob = torch.sigmoid(z)
        r = torch.bernoulli(prob)
        s = prob + (r - prob).detach()   # straight through estimator
        return s

    def construct_multiple_samples(self, mean, std, u_pert, m_samples):
        eps = torch.randn((mean.shape[0], m_samples, mean.shape[1])).to(self.hparams.device)
        eps_corr = torch.randn((u_pert.shape[0], m_samples, u_pert.shape[2], 1)).to(self.hparams.device)
        z = mean.unsqueeze(1) + eps * std.unsqueeze(1) + torch.matmul(u_pert.unsqueeze(1), eps_corr).squeeze()
        classes = torch.multinomial(torch.Tensor([[1 / m_samples] * m_samples]), mean.shape[0], replacement=True).to(self.hparams.device)
        ind = classes.repeat(mean.shape[1], 1).T.unsqueeze(1)
        selected_z = torch.gather(z, 1, ind)

        prob = torch.sigmoid(selected_z)
        sample = prob + (torch.bernoulli(prob) - prob).detach()     # straight through estimator
        return sample, z

    def forward(self, X):
        mu, std, u_pert = self.encode(X)
        s = self.reparameterize(mu, std, u_pert)

        log_likelihood = self.dec(s, X.sign())

        # Compute the KL term
        # To fully understand the KL term, please defers to: 
        s_prime, z_k_samples = self.construct_multiple_samples(mu, std, u_pert, self.hparams.sample_m)
        logqs_r = torch.sum(z_k_samples * s_prime - F.softplus(z_k_samples), dim=2)
        constant = torch.log(torch.tensor(self.hparams.sample_m + 0.)).to(self.hparams.device)
        kl_div_h_p = torch.mean(torch.logsumexp(logqs_r - constant, dim=1) + s.shape[1] * np.log(2.))
        
        loss = -log_likelihood + self.hparams.beta * kl_div_h_p
        
        return {'loss': loss, 'log_likelihood': log_likelihood, 'kl': kl_div_h_p}

    def encode_discrete(self, X):
        mu, _, _ = self.encode(X)
        prob = torch.sigmoid(mu)
        return prob.round()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'lr': [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001],
            'dim_hidden': [300, 400, 500, 600, 700],
            'num_components': [10, 20, 40, 80],
            'num_layers': [0, 1, 2],
            'beta': [1, 2, 3],
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument('--num_features', type=int, default=32,
                            help='num discrete features [%(default)d]')
        parser.add_argument('--batch_size', type=int, default=100,
                            help='batch size [%(default)d]')
        parser.add_argument('--lr', type=float, default=0.005,
                            help='initial learning rate [%(default)g]')

        parser.add_argument('--dim_hidden', type=int, default=500,
                            help='dimension of hidden state [%(default)d]')
        parser.add_argument('--num_layers', type=int, default=0,
                            help='num layers [%(default)d]')
        parser.add_argument('--beta', type=float, default=1,
                            help='beta term (as in beta-VAE) [%(default)g]')
        parser.add_argument('-k', '--rank', type=int, default=10,
                            help='degree of lower rank disturbance [%(default)g]')
        parser.add_argument('-m', '--sample_m', type=int, default=10,
                            help='number of samples used to construct the lower bound of ELBO. [%(default)g]')

        parser.add_argument('--median_threshold', type=bool, default=False,
                            help='num mixture components [%(default)d]')

        return parser

class Encoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
        super().__init__()
        self.ff = FF(dim_input, dim_hidden, dim_output, num_layers,activation='relu', dropout_rate=0.2)

    def forward(self, Y):
        return F.relu(self.ff(Y))

class Decoder(nn.Module):  # As in VDSH, NASH, BMSH
    def __init__(self, dim_encoding, vocab_size):
        super().__init__()
        self.E = nn.Embedding(dim_encoding, vocab_size)
        self.b = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, Z, targets):  # (B x m), (B x V binary)
        scores = Z @ self.E.weight + self.b # B x V
        log_probs = scores.log_softmax(dim=1)
        log_likelihood = (log_probs * targets).sum(1).mean()
        return log_likelihood