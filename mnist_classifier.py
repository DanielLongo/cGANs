import torch
import time
from torch import nn
import torchvision.datasets
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(500, 50)
		# self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		# x = x.view(-1, 320)
		x = x.view(-1, 500)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)

		return F.log_softmax(x, dim=1)

def train(model, num_epochs, lr, train_loader):
	solver = torch.optim.Adam(model.parameters(), lr=lr)
	loss_op = nn.CrossEntropyLoss()  
	for i in range(num_epochs):
		for x,y in train_loader:
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
			solver.zero_grad()
			a = model(x)
			cost = F.nll_loss(a, y)
			cost.backward()
			solver.step()
		print("Epoch:", i, "Cost:", cost)
	return model

def test(model, test_loader):
	model.eval()
	avg_correct = []
	costs = []
	with torch.no_grad():
		for x, y in test_loader:
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
			a = model(x)
			preds = a.max(1, keepdim=True)[1]
			correct = preds.eq(y.view_as(preds)).float().mean()
			cost = F.nll_loss(a, y)
			avg_correct += [correct]
			costs += [cost]
	avg_correct = float(sum(avg_correct)/len(avg_correct))
	avg_cost = float(sum(costs)/len(costs))
	return avg_cost, avg_correct

def main():
	filepath = "./saved_models/"
	filename = "mnist_classifer"
	batch_size = 128
	img_size = 32
	transform = transforms.Compose([
		transforms.Resize(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

	mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
	mnist_test = torchvision.datasets.MNIST('./MNIST_data', train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,  shuffle=True)
	net = Net()
	if torch.cuda.is_available():
		net = net.cuda()
	torch.backends.cudnn.benchmark = True
	net = train(net, 20, .002, train_loader)
	cost, correct = test(net, test_loader)
	print("cost", cost, "correct", correct)
	torch.save(net.state_dict(), filepath + filename + ".pt")

if __name__ == "__main__":
	main()