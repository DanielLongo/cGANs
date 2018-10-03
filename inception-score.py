import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision
from torchvision import transforms
from DCGAN import Discriminator, Generator

# from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def get_preds(x, model):
	preds  = F.softmax(model(x))
	preds= preds.data.cpu().numpy()
	return preds

def inception_score(imgs, model, batch_size, splits=1):
	if torch.cuda.is_available():
		print("Running on a GPU :)")
		dtype = torch.cuda.FloatTensor
		model = model.cuda()
	else:
		print("Running on a CPU :(")
		dtype = torch.FloatTensor
	inception_model = model
	# inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
	# inception_model.eval();

	img_loader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=True)


	N = len(imgs)

	preds = np.zeros((N, 1000))
	for i, (examples,_) in enumerate(img_loader):
		examples = examples.type(dtype)
		if examples.shape[0] != batch_size:
			continue
		preds[i*batch_size:(i*batch_size) + batch_size] = get_preds(examples, inception_model)

	split_scores = []

	for k in range(splits):
		part = preds[k * (N // splits): (k+1) * (N // splits), :]
		py = np.mean(part, axis=0)
		scores = []
		for i in range(part.shape[0]):
			pyx = part[i, :]
			scores.append(entropy(pyx, py))
		split_scores.append(np.exp(np.mean(scores)))

	return np.mean(split_scores), np.std(split_scores)

if __name__ == "__main__":
	img_size = 32
	discriminator = Discriminator()
	discriminator.load_state_dict(torch.load("./transD_mnist.pt"))
	transform = transforms.Compose([
		transforms.Resize(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
	mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)

	score = inception_score(mnist_train, discriminator, 32)
	print(score[0], score[1])
