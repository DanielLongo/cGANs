import torch
from DCGAN import train_gan, Discriminator, Generator
from cDCGAN import ConditionalGenerator
from utils import save_run
from inception_score import inception_score
import torchvision.datasets
import torchvision
from torchvision import transforms

img_size = 32
batch_size = 128
pretrained_generator_filepath = "./saved_models/cG-mnist.pt"

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

generator.deconv1 = pretrained_generator.layer1_input
generator.deconv1.requires_grad = False

if __name__ == "__main__":
    d_filename = "testD"
    g_filename = "testG"
    filename = "test"
    num_epochs = 2
    g_lr = .0002
    d_lr = .0002
    discriminator, generator = train_gan(discriminator, generator, train_loader, num_epochs, batch_size, g_lr, d_lr, dtype, filename_prefix="trans_DCGAN-")
    print("training finished")
    fake_images = []
    for i in range(16):
        fake_images += [generator(generate_noise(batch_size))]
    inception_score = inception_score(fake_images)
    save_run(inception_score, g_lr, num_epochs, discriminator, generator, filename, g_filename, d_filename)
    print("run saved")
