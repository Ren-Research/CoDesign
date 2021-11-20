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
from nasbench.darts.genotypes import get_genotype_from_arch 
import nasbench.darts.utils

import pickle

def get_model():
	print("hello")
	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
			format=log_format, datefmt='%m/%d %I:%M:%S %p')
	
	CIFAR_CLASSES = 10
	
	model_arch = pickle.load(open("./nasbench/model_arch.pickle", "rb"))
	all_model = []
	
	index = 0
	for arch in model_arch:
		if index < 10000:
			genotype = get_genotype_from_arch(arch)
			#genotype = eval("genotypes.%s" % 'DARTS')
			#print(genotype)
			print(index)
		#	for geno in model_genotype:
		#		genotype = eval("genotypes.%s" % 'DARTS')
			model = Network(36, CIFAR_CLASSES, 20, True, genotype)
			print('DARTS', genotype)
			#print(model)
			#utils.load(model, './cifar10_model.pt')
			#torch.save(model, './model/0.pt')
			
			#model2 = torch.load('./model/0.pt')
			all_model.append(model)
			index += 1
	print(all_model)
	return all_model


#get_model()