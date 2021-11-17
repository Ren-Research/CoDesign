#!/usr/bin/env python3

import os
import sys
import glob
import numpy as np
import torch
import logging
import argparse

import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from nasbench.darts.model import NetworkCIFAR as Network
import nasbench.darts.genotypes as genotypes
import nasbench.darts.utils


def get_model():

	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
			format=log_format, datefmt='%m/%d %I:%M:%S %p')
	
	CIFAR_CLASSES = 100
	
	
	genotype = eval("genotypes.%s" % 'DARTS')
	model = Network(36, CIFAR_CLASSES, 20, True, genotype)
	print('DARTS', genotype)
	print(model)
	#utils.load(model, './cifar10_model.pt')
	#torch.save(model, './model/0.pt')
	
	#model2 = torch.load('./model/0.pt')
	return model