import time 
import random
import json
import torch

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

def generate_nosie(batch_size, dim=100):
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
		num_values += [random.uniform(min, max)]

	if len(x) > len(set(x)):
		#not all values unique, try again
		return get_random_params(min, max,  num_values)
	return values
