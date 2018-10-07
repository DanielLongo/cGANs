import math
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3

def inception_score(imgs, use_cuda=None):
	net = inception_v3(pretrained=True).cuda()
	if cuda.is_availiable() and use_cuda != False:
		net = net.cuda()
	elif !(cuda.is_availiable()) and use_cuda == False:
		use_cuda = False
	else:
		print("Cuda not availiabe but use_cuda is True")
		return

	assert(len(imgs.shape) == 5), "Batches of imgs had incorrect number of dimensions. Expected 5"
	scores = []

	for batch in imgs:
		batch = batch.cuda()
		s, _ = net(batch)
	p_yx = F.softmax(torch.cat(scores, 0), 1)
	p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
	KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
	final_score = KL_d.mean()
	return final_score




