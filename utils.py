import time 

def save_run(inception_score, lr, epochs, discriminator, generator, filename, g_filename, d_filename):
	info = [{
		"inception_score" : inception_score,
		"lr" : lr,
		"epochs" : epochs,
		"timestamp" : int(time.time())
		"g_filename" : g_filename,
		"d_filename" : d_filename
	}]

	torch.save(discriminator.state_dict(), d_filename + ".pt")
	torch.save(generator.state_dict(), g_filename + ".pt")
