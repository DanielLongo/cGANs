import time 
import random
import json

def save_run(inception_score, lr, epochs, discriminator, generator, filename, g_filename, d_filename):
	info = {
		"inception_score" : inception_score,
		"lr" : lr,
		"epochs" : epochs,
		"timestamp" : int(time.time())
		"g_filename" : g_filename,
		"d_filename" : d_filename
	}

	info_json = json.dumps(info)
	file = open(filename, "w+")
	json.dump(info_json, file)

	torch.save(discriminator.state_dict(), d_filename + ".pt")
	torch.save(generator.state_dict(), g_filename + ".pt")

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
