import argparse
import torch
import torch.nn as nn

from model.base_model import Base_Model
import utils.entropy as ent
from utils.pytorch_helper import FF, get_init_function

class AMMI(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def define_parameters(self):
        self.entropy = EntropyHelper(self.hparams)
        self.pZ_Y = Posterior(self.hparams.num_features,
                              self.hparams.order_posterior,
                              self.data.vocab_size,
                              self.hparams.num_layers_posterior,
                              self.hparams.dim_hidden)

        if not self.hparams.brute:
            self.qZ = Prior(self.hparams.num_features,
                            self.hparams.order_prior,
                            self.hparams.num_layers,  # Using general num_layers
                            self.hparams.dim_hidden,  # Using general dim_hidden
                            self.hparams.raw_prior)
        
        self.apply(get_init_function(self.hparams.init))
        self.lr_prior = self.hparams.lr if self.hparams.lr_prior < 0 else \
                        self.hparams.lr_prior
    
    def forward(self, Y):
        P_ = self.pZ_Y(Y.sign())
        P = torch.sigmoid(P_)

        Q_ = P_
        hZ_cond = self.entropy.hZ_X(P, Q_)

        if self.hparams.brute:
            hZ = self.entropy.hZ(P)
        else:
            optimizer_prior = torch.optim.Adam(self.qZ.parameters(),
                                               lr=self.lr_prior)
            for _ in range(self.hparams.num_steps_prior):
                optimizer_prior.zero_grad()
                hZ = self.entropy.hZ_X(P.detach(), self.qZ())
                hZ.backward()
                nn.utils.clip_grad_norm_(self.qZ.parameters(),
                                         self.hparams.clip)
                optimizer_prior.step()

            hZ = self.entropy.hZ_X(P, self.qZ())

        loss = hZ_cond - self.hparams.entropy_weight * hZ
        return {'loss': loss, 'hZ_cond': hZ_cond, 'hZ': hZ}
    
    def configure_optimizers(self):
        params = list(self.pZ_Y.parameters())
        return torch.optim.Adam(self.pZ_Y.parameters(), lr=self.hparams.lr)
    
    def configure_gradient_clippers(self):
        clippers = [(self.pZ_Y.parameters(), self.hparams.clip)]
        return clippers
    
    def encode_discrete(self, Y):
        P = torch.sigmoid(self.pZ_Y(Y.sign()))
        encodings = self.entropy.viterbi(P)[0]
        return encodings  # {0,1}^{B x m}
    
    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'lr_prior': [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001],
            'entropy_weight': [1, 1.5, 2, 2.5, 3, 3.5],
            'num_steps_prior': [1, 2, 4],
            'dim_hidden': [8, 12, 16, 20, 24, 28],
            'num_layers': [0, 1, 2],
            'raw_prior': [False, False, False, True],
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument('--num_features', type=int, default=32,
                            help='num discrete features [%(default)d]')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='batch size [%(default)d]')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='initial learning rate [%(default)g]')

        parser.add_argument('--order_posterior', type=int, default=0,
                            help='Markov order of posterior [%(default)d]')
        parser.add_argument('--order_prior', type=int, default=3,
                            help='Markov order of prior [%(default)d]')
        parser.add_argument('--num_layers_posterior', type=int, default=0,
                            help='num layers in posterior [%(default)d]')
        parser.add_argument('--num_steps_prior', type=int, default=4,
                            help='num gradient steps on prior per loss '
                            '[%(default)d]')
        parser.add_argument('--raw_prior', action='store_true',
                            help='raw logit embeddings for prior encoder?')
        parser.add_argument('--lr_prior', type=float, default=-1,
                            help='initial learning rate for prior (same as lr '
                            ' if -1) [%(default)g]')
        parser.add_argument('--entropy_weight', type=float, default=2,
                            help='entropy weight in MI [%(default)g]')
        parser.add_argument('--brute', action='store_true',
                            help='brute-force entropy calculation?')

        parser.add_argument('--dim_hidden', type=int, default=400,
                            help='dimension of hidden state [%(default)d]')
        parser.add_argument('--num_layers', type=int, default=0,
                            help='num layers [%(default)d]')
        parser.add_argument('--beta', type=float, default=1,
                            help='beta term (as in beta-VAE) [%(default)g]')

        parser.add_argument('--median_threshold', type=bool, default=False,
                            help='num mixture components [%(default)d]')

        return parser
    

class EntropyHelper(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.register_buffer('quads',
                            ent.precompute_quads(hparams.order_posterior))
        assert hparams.order_prior >= hparams.order_posterior
        device = torch.device('cuda' if hparams.cuda else 'cpu')
        self.buffs = ent.precompute_buffers(hparams.batch_size,
                                            hparams.order_posterior,
                                            hparams.order_prior,
                                            device)
        if hparams.brute:
            self.register_buffer('I', ent.precompute_I(hparams.num_features,
                                                    hparams.order_posterior))

    def hZ_X(self, P, Q_):
        if len(Q_.size()) == 2:
            Q_ = Q_.repeat(P.size(0), 1, 1)
        return ent.estimate_hZ_X(P, Q_, quads=self.quads, buffers=self.buffs)

    def hZ(self, P):
        return ent.estimate_hZ(P, I=self.I.repeat(P.size(0), 1, 1))

    def viterbi(self, P):
        return ent.compute_viterbi(P, quads=self.quads)


class Posterior(nn.Module):
    def __init__(self, num_features, markov_order, dim_input, num_layers,
                 dim_hidden):
        super(Posterior, self).__init__()
        self.num_features = num_features

        num_logits =  num_features * pow(2, markov_order)
        self.ff = FF(dim_input, dim_hidden, num_logits, num_layers)

    def forward(self, inputs):
        logits = self.ff(inputs).view(inputs.size(0), self.num_features, -1)
        P_ = torch.cat([-logits, logits], dim=2)  # B x m x 2^(o+1)
        return P_


class Prior(nn.Module):
    def __init__(self, num_features, markov_order, num_layers, dim_hidden,
                 raw=False):
        super(Prior, self).__init__()
        self.raw = raw

        if raw:
            self.theta = nn.Embedding(num_features, pow(2, markov_order))
        else:
            self.theta = nn.Embedding(num_features, dim_hidden)
            self.ff = FF(dim_hidden, dim_hidden, pow(2, markov_order),
                         num_layers)

    def forward(self):
        logits = self.theta.weight if self.raw else self.ff(self.theta.weight)
        R_ = torch.cat([-logits, logits], dim=1)  # m x 2^(r+1)
        return R_