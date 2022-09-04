from time import sleep
import torch
import os
import numpy as np
import cv2
from model import *
import sys
import pickle
import collections

def loader(path):
    image = np.asarray(cv2.imread(path)).astype(np.uint8) # [H x W x C, BGR format]
    return image.copy()

def files_inference(weights, data_folder, class_labels, device='cpu'):
	face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	classifier = EmotionNet(5)
	with open(weights, "rb") as weightfile:
		data = pickle.load(weightfile)
		data = collections.OrderedDict(data)
		classifier.load_state_dict(data)
	classifier.eval()
	
	try:
		files = [f for f in os.listdir(data_folder)]
	except:
		print("No such file or directory exists %s" %data_folder)
		return
	
	inference_folder = os.path.join("./data", "inference")
	if not os.path.exists(inference_folder):
		os.makedirs(inference_folder)

	invalid_files = []
	for file in files:
		try:
			sample = loader(os.path.join(data_folder, file))
		except:
			invalid_files.append(os.path.join(data_folder, file))
			continue

		labels = []
		gray = sample.copy()
		if len(gray.shape) == 3 and gray.shape[-1] != 1:
			gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
		faces = face_classifier.detectMultiScale(gray,1.3,5)

		for (x, y, w, h) in faces:
			cv2.rectangle(sample, (x,y), (x+w,y+h), (255,0,0), 2)
			roi = gray[y:y+h, x:x+w]
			roi = cv2.resize(roi, (48,48), interpolation=cv2.INTER_AREA)

			if np.sum([roi]) != 0:
				roi = roi.astype('float')/255
				roi = torch.from_numpy(roi.copy()).unsqueeze(0).unsqueeze(0)
				roi = roi.type(torch.FloatTensor).to(device)
				roi = (roi - 0.5076) / 0.0647
				with torch.no_grad():
					pred = classifier(roi).squeeze()
				_, ind = torch.max(pred, dim=0)
				label = class_labels[ind.item()]
				label_position = (x,y)
				cv2.putText(sample, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
			else:
				cv2.putText(sample, 'No Face Found', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

		cv2.imwrite(os.path.join(inference_folder, file), sample)
	
	if len(invalid_files) > 0:
		print("The following files %d could not be processed:" %(len(invalid_files)))
		for i in range(len(invalid_files)):
			print(invalid_files[i])

def camfeed_inference(weights, class_labels, device='cpu'):
	face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	classifier = EmotionNet(5)
	with open(weights, "rb") as weightfile:
		data = pickle.load(weightfile)
		data = collections.OrderedDict(data)
		classifier.load_state_dict(data)
	classifier.eval()

	flag = False
	try:
		cap = cv2.VideoCapture(0)

		while True:
			# Grab a single frame of video
			ret, frame = cap.read()
			labels = []
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = face_classifier.detectMultiScale(gray,1.3,5)

			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
				roi = gray[y:y+h, x:x+w]
				roi = cv2.resize(roi, (48,48), interpolation=cv2.INTER_AREA)

				if np.sum([roi]) != 0:
					roi = roi.astype('float')/255
					roi = torch.from_numpy(roi.copy()).unsqueeze(0).unsqueeze(0)
					roi = roi.type(torch.FloatTensor).to(device)
					roi = (roi - 0.5076) / 0.0647
					with torch.no_grad():
						pred = classifier(roi).squeeze()
					_, ind = torch.max(pred, dim=0)
					label = class_labels[ind.item()]
					label_position = (x,y)
					cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
				else:
					cv2.putText(frame, 'No Face Found', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

			cv2.imshow('Fast Face Expression', frame)

			if cv2.waitKey(1) == 27: # & 0xFF == ord('q'):
				flag = True
				break

		cap.release()
		cv2.destroyAllWindows()

	except:
		print("Unexpected error!\n")
		if not flag:
			cap.release()
			cv2.destroyAllWindows()
			raise