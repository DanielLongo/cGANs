import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3
import numpy as np
def get_inception_score(imgs, use_cuda=None):
	add_channels = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Grayscale(3),
		transforms.ToTensor()])
	up = nn.Upsample(size=(299, 299), mode='bilinear')
	net = inception_v3(pretrained=True)
#	net.eval()
	if torch.cuda.is_available() and use_cuda != False:
		net = net.cuda()
	elif (torch.cuda.is_available() == False) and use_cuda == False:
		use_cuda = False
	else:
		print("Cuda not availiabe but use_cuda is True")
		return

	batch_size = np.shape(imgs[0])[0] 
	assert(len(np.shape(imgs[0])) == 4), "Batches of imgs had incorrect number of dimensions. Expected 5. Recieved shape: " + str(np.shape(imgs))
	scores = []

	for batch in imgs:
		batch_with_channels = torch.zeros((batch_size,3, 32, 32))
		for i in range(len(batch)):
			img = batch[i,:,:]
			curr_img = img.detach().cpu().numpy().T
			img = add_channels(curr_img).squeeze()
			batch_with_channels[i,:,:] = img
		batch = batch_with_channels
		batch = up(batch)
		s,_ = net(batch)
		scores +=  [s]
	print("scores calculated")
	p_yx = F.softmax(torch.cat(scores, 0), 1)
	p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
	KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
	final_score = KL_d.mean()
	final_score = float(final_score.detach().cpu().numpy())
	print("inception score", final_score)
	return final_score




