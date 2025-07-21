#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import sys
import logging

import torch.distributed as dist


def setup_logger(logpath):
    logfile = 'Net-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpath, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank()==0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())

# logpath='./logs'
# logfile = 'Net-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
# logfile = osp.join(logpath, logfile)
# FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
# log_level = logging.INFO
# logging.basicConfig(level=log_level, format=FORMAT, filename=logfile,filemode = 'w')
# print(logfile)
# print(FORMAT)
