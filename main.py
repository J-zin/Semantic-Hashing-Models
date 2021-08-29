import argparse
import torch
import torch.nn as nn

from model.NASH import NASH
from model.VDSH import VDSH
from model.BMSH import BMSH
from model.WISH import WISH
from model.AMMI import AMMI
from model.corrSH import corrSH

if __name__ == '__main__':
    model_class_mapping = {
        "vdsh": VDSH,
        "nash": NASH,
        "wish": WISH,
        "bmsh": BMSH,
        "ammi": AMMI,
        "corrsh": corrSH,
    }

    model_name = 'corrsh'

    argparser = model_class_mapping[model_name].get_model_specific_argparser()
    hparams = argparser.parse_args()
    model = model_class_mapping[model_name](hparams)


    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        # model.hash_code_generation()

        val_perf, test_perf = model.run_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))