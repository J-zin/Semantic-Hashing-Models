import argparse
import torch
import torch.nn as nn

from model.base_model import Base_Model
from utils.pytorch_helper import FF, get_init_function

class WISH(Base_Model):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def define_parameters(self):
        self.enc = BerEncoder(self.data.vocab_size, self.hparams.dim_hidden,
                              self.hparams.num_features,
                              self.hparams.num_layers)
        self.dec = Decoder(self.hparams.dim_topics, self.data.vocab_size)
        self.logvar_mlp = FF(self.hparams.num_features,   # data dependent noise generator
                             self.hparams.dim_hidden,
                             self.hparams.num_features, 1)
        self.topicBook = nn.Parameter(torch.rand(self.hparams.num_features, self.hparams.dim_topics, requires_grad=True))
        self.I = nn.Parameter(torch.eye(self.hparams.num_features), requires_grad=False)

    def forward(self, X):
        q_x = self.enc(X.sign())
        Z = torch.bernoulli(q_x)

        Z_st = q_x + (Z - q_x).detach()   # straight through estimator

        topic_com = torch.mm(Z_st, self.topicBook)

        log_likelihood = self.dec(topic_com, X.sign())
        kl = torch.mean(torch.sum(q_x * torch.log(2 * q_x + 1e-8) + (1 - q_x) * torch.log(2*(1 - q_x) + 1e-8), dim=-1))
        reg = torch.pow(torch.norm(torch.mm(self.topicBook, self.topicBook.t()) - self.I), 2)
        
        loss = -log_likelihood + self.hparams.beta * kl + self.hparams.alpha * reg

        return {'loss': loss, 'log_likelihood': log_likelihood, 'kl': kl}
    
    def encode_discrete(self, X):
        return self.enc(X.sign()).round()

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
        parser.add_argument('--batch_size', type=int, default=32,
                            help='batch size [%(default)d]')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='initial learning rate [%(default)g]')

        parser.add_argument('--dim_hidden', type=int, default=400,
                            help='dimension of hidden state [%(default)d]')
        parser.add_argument('--num_layers', type=int, default=0,
                            help='num layers [%(default)d]')
        parser.add_argument('--dim_topics', type=int, default=100,
                            help='num layers [%(default)d]')
        parser.add_argument('--beta', type=float, default=1,
                            help='beta term (as in beta-VAE) [%(default)g]')
        parser.add_argument('--alpha', type=float, default=1,
                            help='alpha term (regularization) [%(default)g]')

        parser.add_argument('--median_threshold', type=bool, default=False,
                            help='num mixture components [%(default)d]')

        return parser

class BerEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
        super().__init__()
        self.ff = FF(dim_input, dim_hidden, dim_output, num_layers)

    def forward(self, Y):
        return torch.sigmoid(self.ff(Y))

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