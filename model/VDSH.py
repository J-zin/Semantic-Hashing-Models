import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_model import Base_Model
from utils.pytorch_helper import FF, get_init_function

class VDSH(Base_Model):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def define_parameters(self):
        self.enc = GauEncoder(self.data.vocab_size, self.hparams.dim_hidden,
                              self.hparams.num_features,
                              self.hparams.num_layers)
        self.dec = Decoder(self.hparams.num_features, self.data.vocab_size)

    def forward(self, X):
        q_mu, q_sigma = self.enc(X.sign())
        eps = torch.randn_like(q_mu)
        Z_st = q_mu + q_sigma * eps

        log_likelihood = self.dec(Z_st, X.sign())
        kl = self.compute_kl(q_mu, q_sigma)

        loss = -log_likelihood + self.hparams.beta * kl
        return {'loss': loss, 'log_likelihood': log_likelihood, 'kl': kl}
    
    def compute_kl(self, mu, sigma):
        return torch.mean(-0.5 * torch.sum(1 - mu**2 - sigma**2 + 2*torch.log(sigma + 1e-8), dim=-1))

    def get_median_threshold_binary_code(self, train_loader, eval_loader, device):
        def extract_data(loader):
            encoding_chunks = []
            label_chunks = []
            for (docs, labels) in loader:
                docs = docs.to(device)
                embedding, _ = self.enc(docs.sign())
                encoding_chunks.append(embedding)
                label_chunks.append(labels)

            encoding_mat = torch.cat(encoding_chunks, 0)
            label_mat = torch.cat(label_chunks, 0)
            label_lists = [[j.item() for j in label_mat[i].nonzero()] for i in
                        range(label_mat.size(0))]
            return encoding_mat, label_lists
        
        train_mu, train_y = extract_data(train_loader)
        test_mu, test_y = extract_data(eval_loader)


        mid_val, _ = torch.median(train_mu, dim=0)
        train_b = (train_mu > mid_val).type(torch.FloatTensor).to(device)
        test_b = (test_mu > mid_val).type(torch.FloatTensor).to(device)

        del train_mu
        del test_mu
        return train_b, test_b, train_y, test_y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'lr': [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001],
            'dim_hidden': [300, 400, 500, 600, 700],
            'num_layers': [0, 1, 2],
            'beta': [1, 2, 3],
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument('--num_features', type=int, default=32,
                            help='num discrete features [%(default)d]')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='batch size [%(default)d]')
        parser.add_argument('--lr', type=float, default=0.003,
                            help='initial learning rate [%(default)g]')

        parser.add_argument('--dim_hidden', type=int, default=500,
                            help='dimension of hidden state [%(default)d]')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='num layers [%(default)d]')
        parser.add_argument('--beta', type=float, default=1,
                            help='beta term (as in beta-VAE) [%(default)g]')

        parser.add_argument('--median_threshold', type=bool, default=True,
                            help='num mixture components [%(default)d]')
        
        return parser


class GauEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
        super().__init__()
        self.ff = FF(dim_input, dim_hidden, dim_hidden, num_layers)
        self.mu_enc = nn.Linear(dim_hidden, dim_output)
        self.sigma_enc = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        hidden = F.relu(self.ff(x))
        mu = self.mu_enc(hidden)
        sigma = F.softplus(self.sigma_enc(hidden))
        return mu, sigma

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