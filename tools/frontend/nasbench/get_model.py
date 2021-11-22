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
from nasbench.darts.utils import flops_counter

import pickle

#def get_model():
#	print("hello")
#	log_format = '%(asctime)s %(message)s'
#	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#			format=log_format, datefmt='%m/%d %I:%M:%S %p')
#	
#	CIFAR_CLASSES = 10
#	
#	model_arch = pickle.load(open("./nasbench/model_arch.pickle", "rb"))
#	flops_arr = pickle.load(open("./nasbench/flops_arr.pickle", "rb"))
#	print(len(flops_arr))
#	all_model = []
#	
#	index = 0
#	#flops_arr = []
#	for i in range(len(flops_arr), len(model_arch)):
#		if index < 10000:
#			arch = model_arch[i]
#			genotype = get_genotype_from_arch(arch)
#			#genotype = eval("genotypes.%s" % 'DARTS')
#			#print(genotype)
#			print(i)
#		#	for geno in model_genotype:
#		#		genotype = eval("genotypes.%s" % 'DARTS')
#			model = Network(36, CIFAR_CLASSES, 20, True, genotype)
#			flops, params = flops_counter(model, (1, 3, 16, 16))
#			flops_arr.append(flops)
#			#print('DARTS', genotype)
#			#print(model)
#			#utils.load(model, './cifar10_model.pt')
#			#torch.save(model, './model/0.pt')
#			
#			#model2 = torch.load('./model/0.pt')
#			all_model.append(model)
#			index += 1
#			
#			if index % 1000 == 0 or index == len(model_arch)-1:
#				pickle.dump(flops_arr, open("./nasbench/flops_arr.pickle", "wb"))
#				print(flops_arr)
#				print(len(flops_arr))
#				
#	print(all_model)
#	return all_model


#get_model()


def get_model():
	print("hello")
	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
			format=log_format, datefmt='%m/%d %I:%M:%S %p')
	
	CIFAR_CLASSES = 10
	
	model_arch = pickle.load(open("./nasbench/model_arch.pickle", "rb"))
	select_index = pickle.load(open("./nasbench/model_index.pickle", "rb"))
	print(len(select_index))
	all_model = []
	
	for i in range(100):
		index = select_index[i]
		arch = model_arch[index]
		genotype = get_genotype_from_arch(arch)
		print(i)
		model = Network(36, CIFAR_CLASSES, 20, True, genotype)
		all_model.append(model)
				
	return all_model