import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3
import numpy as np
# def get_inception_score(imgs, use_cuda=None):
# 	add_channels = transforms.Compose([
# 		transforms.ToPILImage(),
# 		transforms.Grayscale(3),
# 		transforms.ToTensor()])
# 	up = nn.Upsample(size=(299, 299), mode='bilinear')
# 	net = inception_v3(pretrained=True)
# 	net.eval()
# 	if torch.cuda.is_available() and use_cuda != False:
# 		net = net.cuda()
# 	elif (torch.cuda.is_available() == False) and use_cuda == False:
# 		use_cuda = False
# 	else:
# 		print("Cuda not availiabe but use_cuda is True")
# 		return

# 	batch_size = np.shape(imgs[0])[0] 
# 	assert(len(np.shape(imgs[0])) == 4), "Batches of imgs had incorrect number of dimensions. Expected 5. Recieved shape: " + str(np.shape(imgs))
# 	# scores = []
# 	n = batch_size * np.shape(imgs)[0]
# 	preds =  np.zeros((n,1000))

# 	for batch in imgs:
# 		batch_with_channels = torch.zeros((batch_size,3, 32, 32))
# 		for i in range(batch_size):
# 			# img = batch[i,:,:]
# 			curr_img = batch[i]
# 			curr_img = curr_img.detach().cpu().numpy().T
# 			curr_img = add_channels(curr_img).squeeze()
# 			# batch_with_channels[i,:,:] = img
# 			batch_with_channels[i] = curr_img
# 		batch = batch_with_channels
# 		batch = up(batch)
# 		s = net(batch)
# 		s = F.softmax(s).data.cpu().numpy()

# 		# scores += [s]
# 		preds[i+batch_size:(i+1)*batch_size] = s

# 	splits = 10
# 	for k in range(splits):
# 		part = preds[k * (N // splits): (k+1) * (N // splits), :]
# 		py = np.mean(part, axis=0)
# 		scores = []

# 	for i in range(part.shape[0]):
# 		pyx = part[i, :]
# 		scores.append(entropy(pyx, py))
# 	split_scores.append(np.exp(np.mean(scores)))

# 	return np.mean(split_scores)#, np.std(split_scores)
# 	# print("scores", scores)
# 	# p_yx = F.softmax(torch.cat(scores, 0), 1)
# 	# p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
# 	# KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
# 	# final_score = KL_d.mean()
# 	# final_score = float(final_score.detach().cpu().numpy())
# 	# print("inception score", final_score)
# 	# return final_score

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def get_inception_score(imgs, cuda=True, resize=True, splits=1):
	"""Computes the inception score of the generated images imgs
	# imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
	cuda -- whether or not to run on GPU
	batch_size -- batch size for feeding into Inception v3
	splits -- number of splits
	"""
	# print(len(imgs))
	# print(imgs[])
	batch_size = ((imgs[0]).shape)[0]
	N = len(imgs) * batch_size
	# Set up dtype
	if cuda:
		dtype = torch.cuda.FloatTensor
	else:
		if torch.cuda.is_available():
			print("WARNING: You have a CUDA device, so you should probably set cuda=True")
		dtype = torch.FloatTensor

	# Set up dataloader
	# dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

	# Load inception model
	inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
	inception_model.eval();
	up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
	add_channels = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Grayscale(3),
		transforms.ToTensor()])
	def get_pred(x):
		x = up(x)
		x = inception_model(x)
		return F.softmax(x).data.cpu().numpy()

	# Get predictions
	preds = np.zeros((N, 1000))

	# for i, batch in enumerate(dataloader, 0):
	for i, batch in enumerate(imgs):
		batch_with_channels = torch.zeros((batch_size,3, 32, 32))
		for i in range(batch_size):
			curr_img = batch[i]
			curr_img = curr_img.detach().cpu().numpy().T
			curr_img = add_channels(curr_img).squeeze()
			batch_with_channels[i] = curr_img
		batch = batch_with_channels
		batch = batch.type(dtype)
		batchv = Variable(batch)
		batch_size_i = batch.size()[0]

		preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

	# Now compute the mean kl-div
	split_scores = []

	for k in range(splits):
		part = preds[k * (N // splits): (k+1) * (N // splits), :]
		py = np.mean(part, axis=0)
		scores = []
		for i in range(part.shape[0]):
			pyx = part[i, :]
			print(pyx) #all zeros
			# print(py)
			scores.append(entropy(pyx, py))
		split_scores.append(np.exp(np.mean(scores)))
	print("mean score", np.mean(split_scores))
	return np.mean(split_scores) #, np.std(split_scores)
