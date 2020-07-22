from time import sleep
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import *
import sys

def files_inference(weights, device='cpu'):
	pass

def video_inference(weights, device='cpu'):
	pass

def camfeed_inference(weights, device='cpu'):
	face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	classifier = FFENet()
	classifier.load_state_dict(torch.load(weights, map_location=device))
	classifier.eval()

	class_labels = ['Neutral', 'Happy', 'Surprised', 'Sad', 'Angry']
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
				roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

				if np.sum([roi]) != 0:
					roi = roi.astype('float')/127.5 - 1
					roi = np.transpose(roi, (2,0,1))[::-1, ...]
					roi = torch.from_numpy(roi.copy()).unsqueeze(0)
					roi = roi.type(torch.FloatTensor).to(device)
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
