#! /usr/bin/env python

"""
Run face expression detection locally on your system in several modes:
1. On a library of images.
2. On a video stream from your webcam, in real time.
3. On a saved video file (currently unimplemented).
"""

import argparse
import os
import json
import torch
from torch.hub import download_url_to_file
from frontend import *

argparser = argparse.ArgumentParser(
	"Run face expression inference on files, saved video or camera feed")

argparser.add_argument(
	'-m',
	'--mode',
	type	= int,
	help	= "Mode of inference (1, 2 or 3; See README)")

argparser.add_argument(
	'-w',
	'--weights',
	type 	= str,
	default = 'weights/weights.pkl',
	help 	= "Path to EmotionNet weights file")

argparser.add_argument(
	'-d',
	'--device',
	type 	= str,
	default = "cpu",
	help	= "Compute device - 'cuda' or 'cpu'")

argparser.add_argument(
	'-c',
	'--config',
	type	= str,
	default = "config.json",
	help	= "Path to config.json file")

PRETRAINED_WEIGHTS_URL = "https://github.com/codedev99/fast-face-exp/releases/download/v0.3/newenet_paperv3_exp1_net2_5emo.pkl"

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

	with open(args.config) as config_buffer:
		config = json.loads(config_buffer.read())

	if args.mode == 1:
		files_inference(args.weights, config["inference"]["files"]["data_folder"],
							config["inference"]["emotionnet"]["class_labels"], device)
	elif args.mode == 2:
		camfeed_inference(args.weights, config["inference"]["emotionnet"]["class_labels"], device)
	elif args.mode == 3:
		print("Not implemented yet. Please wait for version update.")
	else:
		raise ModeError

if __name__ == '__main__':
	args = argparser.parse_args()
	main(args)