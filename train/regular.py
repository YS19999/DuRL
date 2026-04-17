import os
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from termcolor import colored

from dataset.parallel_sampler import ParallelSampler, ParallelSampler_Test
from train.utils import named_grad_param, grad_param, get_norm


def train(train_data, val_data, model, args):
    '''
        Train the model
        Use val_data to do early stopping
    '''
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
                                      os.path.curdir,
                                      "saved-runs",
                                      args.dataset + '_' + str(args.way) + '_' + str(args.shot)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    sub_cycle = 0
    best_path = None

    opt = torch.optim.Adam(grad_param(model, ['ebd', 'clf']), lr=args.lr)
    optD = torch.optim.Adam(grad_param(model, ['adv']), lr=args.d_lr)

    print("{}, Start training".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)

    train_gen = ParallelSampler(train_data, args, args.train_episodes)
    val_gen = ParallelSampler_Test(val_data, args, args.val_episodes)

    for ep in range(args.train_epochs):
        sampled_tasks = train_gen.get_epoch()

        grad = {'clf': [], 'ebd': [], 'adv': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes, ncols=80, leave=False, desc=colored('Training on train', 'yellow'))

        loss = []
        d_acc = 0.0
        for task in sampled_tasks:
            if task is None:
                break
            d_acc1, loss1 = train_one(task, model, opt, optD, args, grad)
            d_acc += d_acc1
            loss.append(loss1)

        d_acc = d_acc / args.train_episodes

        # print("---------------ep:" + str(ep) + " d_acc:" + str(d_acc) + "-----------")

        # Evaluate validation accuracy
        cur_acc, cur_std = test(val_data, model, args, args.val_episodes, False,
                                val_gen.get_epoch())
        print(("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f}, "
               "{:s} {:s}{:>7.4f}, {:s}{:>7.4f}").format(
               datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
               "ep", ep,
               colored("val  ", "cyan"),
               colored("acc:", "blue"), cur_acc, cur_std, colored("d_acc", "red"), d_acc,
               colored("train stats", "cyan"),
               colored("ebd_grad:", "blue"), np.mean(np.array(grad['ebd'])),
               colored("clf_grad:", "blue"), np.mean(np.array(grad['clf'])),
               ), flush=True)

        # Update the current best model if val acc is better
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_path = os.path.join(out_dir, str(ep))

            # save current model
            print("{}, Save cur best model to {}".format(
                datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
                best_path))

            torch.save(model['ebd'].state_dict(), best_path + '.ebd')
            torch.save(model['clf'].state_dict(), best_path + '.clf')
            torch.save(model['adv'].state_dict(), best_path + '.adv')

            sub_cycle = 0
        else:
            sub_cycle += 1

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

    print("{}, End of training. Restore the best weights".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')),
          flush=True)

    # restore the best saved model
    model['ebd'].load_state_dict(torch.load(best_path + '.ebd', weights_only=True))
    model['clf'].load_state_dict(torch.load(best_path + '.clf', weights_only=True))
    model['adv'].load_state_dict(torch.load(best_path + '.adv', weights_only=True))

    return


def train_one(task, model, opt, optD, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['ebd'].train()
    model['clf'].train()
    model['adv'].train()

    support1, query1, support2, query2 = task

    # Embedding the document
    XS1 = model['ebd'](support1, flag='support')
    YS1 = support1['label']

    XQ1 = model['ebd'](query1)
    YQ1 = query1['label']

    XS2 = model['ebd'](support2, flag='support')
    YS2 = support2['label']

    XQ2 = model['ebd'](query2)
    YQ2 = query2['label']

    all_x1 = torch.cat([XS1, XQ1], dim=0)
    label1 = torch.zeros(all_x1.shape[0], dtype=torch.long, device=all_x1.device)

    all_x2 = torch.cat([XS2, XQ2], dim=0)
    label2 = torch.ones(all_x2.shape[0], dtype=torch.long, device=all_x2.device)

    optD.zero_grad()

    logits1 = model["adv"](all_x1)
    logits2 = model["adv"](all_x2)
    d_acc = (torch.mean((torch.argmax(logits1, dim=1) == label1).float()).item() + torch.mean((torch.argmax(logits2, dim=1) == label2).float()).item()) / 2.
    adv_loss = F.cross_entropy(logits1, label1) + F.cross_entropy(logits2, label2)
    adv_loss.backward(retain_graph=True)
    grad["adv"].append(get_norm(model["adv"]))
    optD.step()

    opt.zero_grad()

    logits1 = model["adv"](all_x1)
    logits2 = model["adv"](all_x2)
    adv_loss = F.cross_entropy(logits1, label1) + F.cross_entropy(logits2, label2)

    # Apply the classifier
    _, loss1 = model['clf'](XS1, YS1, XQ1, YQ1)
    _, loss2 = model['clf'](XS2, YS2, XQ2, YQ2)
    loss = loss1 + loss2 - adv_loss

    if loss is not None:
        loss.backward()

    if torch.isnan(loss):
        # do not update the parameters if the gradient is nan
        # print("NAN detected")
        # print(model['clf'].lam, model['clf'].alpha, model['clf'].beta)
        return

    if args.clip_grad is not None:
        nn.utils.clip_grad_value_(grad_param(model, ['ebd', 'clf']),
                                  args.clip_grad)

    grad['clf'].append(get_norm(model['clf']))
    grad['ebd'].append(get_norm(model['ebd']))

    opt.step()
    return d_acc, loss.item()


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['ebd'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler_Test(test_data, args, num_episodes).get_epoch()

    acc = []
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))

    for task in sampled_tasks:
        acc.append(test_one(task, model, args))

    acc = np.array(acc)

    if verbose:
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
                colored("acc mean", "blue"),
                np.mean(acc),
                colored("std", "blue"),
                np.std(acc),
                ), flush=True)

    return np.mean(acc), np.std(acc)


def test_one(task, model, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    support, query = task

    # Embedding the document
    XS = model['ebd'](support, flag='support')
    YS = support['label']

    XQ = model['ebd'](query)
    YQ = query['label']

    # Apply the classifier
    acc, _ = model['clf'](XS, YS, XQ, YQ)

    return acc
