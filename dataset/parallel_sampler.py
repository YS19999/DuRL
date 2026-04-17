import time
import datetime
from copy import deepcopy, copy
from multiprocessing import Process, Queue, cpu_count

import torch
import numpy as np

import dataset.utils as utils
import dataset.stats as stats

class ParallelSampler():

    def __init__(self, data, args, num_episodes=None):
        self.data = data
        self.args = args
        self.num_episodes = num_episodes

        self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        for y in self.all_classes:
            self.idx_list.append(np.squeeze(np.argwhere(self.data['label'] == y)))

        self.count = 0
        self.done_queue = Queue()

        # self.num_cores = cpu_count() if args.n_workers is 0 else args.n_workers
        self.num_cores = 1

        self.p_list = []
        for i in range(self.num_cores):
            self.p_list.append(
                Process(target=self.worker, args=(self.done_queue,)))

        for i in range(self.num_cores):
            self.p_list[i].start()

    def get_epoch(self):

        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support1, query1, support2, query2 = self.done_queue.get()

            # convert to torch.tensor
            support1 = utils.to_tensor(support1, self.args.cuda, ['raw'])
            query1 = utils.to_tensor(query1, self.args.cuda, ['raw'])
            support2 = utils.to_tensor(support2, self.args.cuda, ['raw'])
            query2 = utils.to_tensor(query2, self.args.cuda, ['raw'])

            support1['is_support'] = True
            query1['is_support'] = False
            support2['is_support'] = True
            query2['is_support'] = False

            yield support1, query1, support2, query2

    def worker(self, done_queue):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > 100:
                time.sleep(1)
                continue

            # sample ways
            classes = np.random.permutation(self.num_classes)
            task_classes1 = classes[:self.args.way]
            task_classes2 = classes[self.args.way:self.args.way + self.args.way]

            # sample examples
            support_idx1, query_idx1 = [], []
            for y in task_classes1:
                tmp = np.random.permutation(len(self.idx_list[y]))
                support_idx1.append(
                    self.idx_list[y][tmp[:self.args.shot]])
                query_idx1.append(
                    self.idx_list[y][tmp[self.args.shot:self.args.shot + self.args.query]])

            support_idx2, query_idx2 = [], []
            for y in task_classes2:
                tmp = np.random.permutation(len(self.idx_list[y]))
                support_idx2.append(self.idx_list[y][tmp[:self.args.shot]])
                query_idx2.append(self.idx_list[y][tmp[self.args.shot:self.args.shot + self.args.query]])

            support_idx1 = np.concatenate(support_idx1)
            query_idx1 = np.concatenate(query_idx1)
            support_idx2 = np.concatenate(support_idx2)
            query_idx2 = np.concatenate(query_idx2)

            # aggregate examples
            max_support_len1 = np.max(self.data['text_len'][support_idx1])
            max_query_len1 = np.max(self.data['text_len'][query_idx1])
            max_support_len2 = np.max(self.data['text_len'][support_idx2])
            max_query_len2 = np.max(self.data['text_len'][query_idx2])

            support1 = utils.select_subset(self.data, {}, ['text', 'text_len', 'attn_mask', 'label'], support_idx1, max_support_len1)
            query1 = utils.select_subset(self.data, {}, ['text', 'text_len', 'attn_mask', 'label'], query_idx1, max_query_len1)
            support2 = utils.select_subset(self.data, {}, ['text', 'text_len', 'attn_mask', 'label'], support_idx2, max_support_len2)
            query2 = utils.select_subset(self.data, {}, ['text', 'text_len', 'attn_mask', 'label'], query_idx2, max_query_len2)

            done_queue.put((support1, query1, support2, query2))

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue

class ParallelSampler_Test():

    def __init__(self, data, args, num_episodes=None):
        self.data = data
        self.args = args
        self.num_episodes = num_episodes

        self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        for y in self.all_classes:
            self.idx_list.append(
                np.squeeze(np.argwhere(self.data['label'] == y)))

        self.count = 0
        self.done_queue = Queue()

        # self.num_cores = cpu_count() if args.n_workers is 0 else args.n_workers
        self.num_cores = 1

        self.p_list = []
        for i in range(self.num_cores):
            self.p_list.append(
                Process(target=self.worker, args=(self.done_queue,)))

        for i in range(self.num_cores):
            self.p_list[i].start()

    def get_epoch(self):
        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support, query = self.done_queue.get()

            # convert to torch.tensor
            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])

            support['is_support'] = True
            query['is_support'] = False

            yield support, query

    def worker(self, done_queue):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > 100:
                time.sleep(1)
                continue
            # sample ways
            sampled_classes = np.random.permutation(self.num_classes)[:self.args.way]

            # sample examples
            support_idx, query_idx = [], []
            for y in sampled_classes:
                tmp = np.random.permutation(len(self.idx_list[y]))
                support_idx.append(
                    self.idx_list[y][tmp[:self.args.shot]])
                query_idx.append(
                    self.idx_list[y][
                        tmp[self.args.shot:self.args.shot + self.args.query]])

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)
            if self.args.mode == 'finetune' and len(query_idx) == 0:
                query_idx = support_idx

            # aggregate examples
            max_support_len = np.max(self.data['text_len'][support_idx])
            max_query_len = np.max(self.data['text_len'][query_idx])

            support = utils.select_subset(self.data, {}, ['text', 'text_len', 'attn_mask', 'label'],
                                          support_idx, max_support_len)
            query = utils.select_subset(self.data, {}, ['text', 'text_len', 'attn_mask', 'label'],
                                        query_idx, max_query_len)

            done_queue.put((support, query))

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue

