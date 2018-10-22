import time 
import random
import json
import torch
import os
from mnist_classifier import Net
def save_run(inception_score, lr, epochs, discriminator, generator, filename, g_filename, d_filename):
	models_filepath = "./saved_models/"
	runs_filepath = "./saved_runs/"
	info = {
		"inception_score" : inception_score,
		"lr" : lr,
		"epochs" : epochs,
		"timestamp" : int(time.time()),
		"g_filename" : models_filepath + g_filename + ".pt",
		"d_filename" : models_filepath + d_filename + ".pt"
	}

	info_json = json.dumps(info)
	file = open(runs_filepath + filename + ".json", "w+")
	json.dump(info_json,  file)

	torch.save(discriminator.state_dict(), models_filepath + d_filename + ".pt")
	torch.save(generator.state_dict(), models_filepath + g_filename + ".pt")

	print("Run saved")
	return info

def read_saved_run(filename, filepath="./saved_runs/"):
	filename = filepath + filename
	with open(filename + ".json", "r") as file:
		data = json.load(file) #reads to string
		data = json.loads(data) #reads string to dict
	return data

def purge_poor_runs(filenames, path, purge_all=False):
	if len(filenames) == 0 and purge_all == False:
		print("print no files to purge")
		return
	elif purge_all == True:
		filenames = os.listdir(path)
	max_inception = 0
	argmax_inception = ""
	for file in filenames:
		cur_stats = read_saved_run(file.split(".json")[0])
		if cur_stats["inception_score"] > max_inception:
			argmax_inception = file

	for file in filenames:
		if file == argmax_inception:
			continue
		os.remove(path + file)

	print("runs purged")


def generate_noise(batch_size, dim=100):
	noise = torch.randn(batch_size, dim, 1, 1)
	return noise

def create_images(generator, batch_size, num_batches):
	images = []
	for i in range(num_batches):
		z = generate_nosie(batch_size)
		a = generator(z)
		images += [a]
	return images


def get_random_params(min, max, num_values):
	values = []
	for i in range(num_values):
		value = random.uniform(min, max)
		value = float("%.6f" % value)
		values += [value]

	if len(values) > len(set(values)):
		#not all values unique, try again
		return get_random_params(min, max,  num_values)
	return values

def get_mnist_classifer(filepath="./saved_models/mnist_classifer.pt"):
	net = Net()
	net.load_state_dict(torch.load(filepath))
	return net