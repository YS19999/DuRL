import torch
import datetime

from embedding.cxtebd import CXTEBD

from embedding.avg import AVG


def get_embedding(args):
    print("{}, Building embedding: {}".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), args.embedding), flush=True)

    ebd = CXTEBD(args, return_seq=(args.embedding != 'ebd'))

    if args.embedding == 'avg':
        model = AVG(ebd, args)  # using bert representation directly
    elif args.embedding == 'ebd':
        model = ebd  # using bert representation directly
    else:
        model = None

    print("{}, Building embedding".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)

    if args.snapshot != '':
        # load pretrained models
        print("{}, Loading pretrained embedding from {}".format(
            datetime.datetime.now().strftime('%0y/%0m/%0d %H:%M:%S'),
            args.snapshot + '.ebd'
            ))
        model.load_state_dict(torch.load(args.snapshot + '.ebd'))

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
