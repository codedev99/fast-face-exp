#! /usr/bin/env python

import argparse
import os
import torch
from torch.hub import download_url_to_file
from frontend import *

argparser = argparse.ArgumentParser(
	"Run face expression inference on files, saved video or camera feed")

argparser.add_argument(
	'-m',
	'--mode',
	type	= int,
	help	= "Mode of inference (See README)")

argparser.add_argument(
	'-w',
	'--weights',
	type 	= str,
	default = 'weights/weights.pth',
	help 	= "Path to FFENet weights file")

argparser.add_argument(
	'-d',
	'--device',
	type 	= str,
	default = "cpu",
	help	= "Compute device - 'cuda' or 'cpu'")

PRETRAINED_WEIGHTS_URL = "https://github.com/codedev99/fast-face-exp/releases/download/v0.2-alpha/ffenet-22-07-2020.pth"

class ModeError(Exception):
	def __init__(self):
		message = "The mode of inference is invalid. Please use an acceptable mode (see README)"
		super(ModeError, self).__init__(message)

def main(argv):
	if not os.path.exists(args.weights):
		download_url_to_file(PRETRAINED_WEIGHTS_URL, args.weights, progress=True)

	device = args.device
	if not torch.cuda.is_available():
		device = 'cpu'



	if args.mode == 1:
		pass
	elif args.mode == 2:
		pass
	elif args.mode == 3:
		camfeed_inference(args.weights, device)
	else:
		raise ModeError

if __name__ == '__main__':
	args = argparser.parse_args()
	main(args)