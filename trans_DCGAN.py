import torch
from DCGAN import train_gan, Discriminator, Generator
from cDCGAN import ConditionalGenerator
from utils import save_run, generate_noise, read_saved_run, get_random_params, purge_poor_runs, get_mnist_classifer
from inception_score_mnist import get_inception_score
import torchvision.datasets
import torchvision
from torchvision import transforms
import numpy as np
import time

img_size = 32
batch_size = 128
pretrained_generator_filepath = "./saved_models/cG-mnist.pt"
pretrained_discriminator_filepath = "./saved_models/D_mnist.pt"

if torch.cuda.is_available():
    print("Running On GPU :)")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    use_cuda = True
else:
    print("Running On CPU :(")
    print("This may take a while")
    use_cuda = False
    dtype = torch.FloatTensor

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# mnist_train = torchvision.datasets.EMNIST('./EMNIST_data', train=True, download=True, transform=transform, split="letters")
mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
# mnist_test = torchvision.datasets.EMNIST('./EMNIST_data', train=False, download=True, transform=transform, split="letters")
# mnist_test = torchvision.datasets.EMNIST('./EMNIST_data', train=False, download=True, transform=transform, split="letters")
# test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,  shuffle=True)

pretrained_generator = ConditionalGenerator()
pretrained_generator.load_state_dict(torch.load(pretrained_generator_filepath))

generator = Generator()
discriminator = Discriminator()
pretrained_discriminator = Discriminator()
pretrained_discriminator.load_state_dict(torch.load(pretrained_discriminator_filepath))


generator.deconv1 = pretrained_generator.input_layer1
# generator.deconv1.requires_grad = False
generator.deconv2 = pretrained_generator.input_layer2
# generator.deconv2.requires_grad = False


if __name__ == "__main__":
    d_filename = "testD"
    g_filename = "testG"
    filename = "control"
    filenames = []
    num_epochs = 10
    random_lrs = get_random_params(.00002, .0002, 50)
    run_stats = []
    for lr in random_lrs:
        print('lr', lr)
        cur_filename_info = str(lr) + "-" + str(num_epochs) + "-" + str(int(time.time()))
        cur_filename = filename + "-" + cur_filename_info 
        filenames += [cur_filename]
        cur_g_filename = g_filename + "-" + cur_filename_info
        cur_d_filename = d_filename + "-" + cur_filename_info
        discriminator, generator = train_gan(discriminator, generator, train_loader, num_epochs, batch_size, lr, lr, dtype, filename_prefix="trans_DCGAN-", save_images=False)
        # discriminator, generator = train_gan(discriminator, generator, train_loader, num_epochs, batch_size, lr*.01, lr, dtype, filename_prefix="trans_DCGAN-", save_images=False)
        # discriminator, generator = train_gan(pretrained_discriminator, generator, train_loader, num_epochs, batch_size, lr, lr, dtype, filename_prefix="trans_DCGAN-", save_images=False)
        fake_images = []
        for i in range(16):
            fake_images += [generator(generate_noise(4))]
        inception_score = get_inception_score(fake_images)
        print(inception_score)
        stats = save_run(inception_score, lr, num_epochs, discriminator, generator, cur_filename, cur_g_filename, cur_d_filename)
        run_stats += [stats]
    print(run_stats)
    purge_poor_runs("./saved_runs/", purge_all=True, start_with=[filename])
    # purge_poor_runs(filenames, "./saved_runs/")
