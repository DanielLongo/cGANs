import torch
from torch import nn
import torchvision.datasets
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
batch_size = 128
img_size = 32
plt.rcParams['image.cmap'] = 'gray'
discriminator_filename = "test_d"
generator_filename = "test_g"

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
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
mnist_test = torchvision.datasets.MNIST('./MNIST_data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,  shuffle=True)
    
def generate_nosie(batch_size, dim=100):
    noise = torch.randn(batch_size, dim, 1, 1)
    return noise

    model = nn.Sequential(
        nn.Conv2d(1, 32, [5,5], stride=[1,1]),
        nn.LeakyReLU(negative_slope=.01),
        nn.MaxPool2d([2,2], stride=[2,2]),
        nn.Conv2d(32, 64, [5,5], stride=[1,1]),
        nn.LeakyReLU(negative_slope=.01),
        nn.MaxPool2d([2,2], stride=[2,2]),
        Flatten(),
        nn.Linear((4*4*64), (4*4*64)), 
        nn.LeakyReLU(negative_slope=.01),
        nn.Linear((4*4*64), 1)
    )
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, [5,5], stride=[1,1]),
            nn.LeakyReLU(negative_slope=.01),
            nn.MaxPool2d([2,2], stride=[2,2]))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, [5,5], stride=[1,1]),
            nn.LeakyReLU(negative_slope=.01),
            nn.MaxPool2d([2,2], stride=[2,2]))
        self.fc1 = nn.Sequential(
            nn.Linear((64*6*6), (64*6*6))
            nn.LeakyReLU(negative_slope=.01))
        self.fc2 = nn.Linear((65*6*6), 1)

    def forward(x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, (7*7*128)),
        nn.ReLU(),
        nn.BatchNorm1d(7*7*128),
        Unflatten(),
        nn.ConvTranspose2d(128, 64, [4,4], stride=[2,2], padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 1, [4,4], stride=[2,2], padding=1),
        nn.Tanh(),
        Flatten()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(100, 256, [4,4], stride=[1,1]),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, [4,4], stride=[2,2], padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, [4,4], stride=[2,2], padding=1),
            nn.BatchNorm2d(128))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, [4,4], stride=[2,2], padding=1),
            nn.Tanh())

    def forward(self, input, labels):
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out


    def weight_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

def create_optimizer(model, lr=.01, betas=None):
    if betas == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    return optimizer

def save_images(generator, epoch, i):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=.05, hspace=.05)
    z = generate_nosie(100)
    images_fake = generator(z)
    images_fake = images_fake.data.data.cpu().numpy()
    for img_num, sample in enumerate(images_fake):
        ax = plt.subplot(gs[img_num])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(32, 32), cmap='Greys_r')

    filename = "DCGAN-" + str(epoch) + "-" + str(i) 
    plt.savefig("./generated_images/" + filename, bbox_inches="tight" )
    plt.close(fig)


def train_gan(discriminator, generator, image_loader, num_epochs, batch_size, lr):
    iters = 0
    d_optimizer = create_optimizer(discriminator, lr=lr, betas=(.5, .999))
    g_optimizer = create_optimizer(generator, lr=lr, betas=(.5, .999))

    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if x.shape[0] != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)
            logits_real = D(2* (real_data - 0.5)).type(dtype)

            z = generate_nosie(batch_size)
            fake_images = generator(z)
            g_result = discriminator(fake_images)
            g_cost = nn.BCELoss(fake_images, torch.ones(batch_size))
            g_cost.backward()
            g_optimizer.step()
            g_optimizer.zero_grad()

            z = generate_nosie(batch_size)
            fake_images = generator(z)
            d_spred_fake = discriminator(fake_images)
            d_cost_fake = nn.BCELoss(fake_images, torch.zeros(batch_size))
            d_spred_real = generator_loss(x)
            d_cost_real = nn.BCELoss(d_spred_real, torch.ones(batch_size))
            d_cost = d_cost_real + d_cost_fake
            d_cost.backward()
            d_optimizer.step()
            iters += 1
        save_images(discriminator)
    return discriminator, generator

