import torch
from classifier.durl import Classifier
from classifier.adv import Adversarial

from dataset.utils import tprint


def get_classifier(ebd_dim, args):

    tprint("Building classifier")

    model = Classifier(ebd_dim, args)
    modelD = Adversarial(ebd_dim)

    if args.snapshot != '':
        # load pretrained models
        tprint("Loading pretrained classifier from {}".format(
            args.snapshot + '.clf'
            ))
        model.load_state_dict(torch.load(args.snapshot + '.clf'))

    if args.cuda != -1:
        return model.cuda(args.cuda), modelD.cuda(args.cuda)
    else:
        return model, modelD
