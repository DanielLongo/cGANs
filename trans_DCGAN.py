import torch
# from DCGAN import train_gan
from DCGAN import train_gan, Discriminator, Generator
# from DCGAN import Generator
from cDCGAN import ConditionalGenerator
import torchvision.datasets
import torchvision
from torchvision import transforms

img_size = 32
batch_size = 128
pretrained_generator_filepath = "cG-mnist.pt"

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
d_lr = .0002
g_lr = d_lr
# g_lr = d_lr * .01
if __name__ == "__main__":
    discriminator_filename = "transD_mnist"
    generator_filename = "transG_mnist"
    discriminator, generator = train_gan(discriminator, generator, train_loader, 20, batch_size, g_lr, d_lr, dtype, filename_prefix="trans_DCGAN-")
    torch.save(generator.state_dict(), generator_filename + ".pt")
    torch.save(discriminator.state_dict(), discriminator_filename + ".pt")
