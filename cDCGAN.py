import torch
from torch import nn
import torchvision.datasets
import numpy as np
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.gridspec as gridspec
batch_size = 128
plt.rcParams['image.cmap'] = 'gray'
use_cuda = False

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True

transform = torchvision.transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)
mnist_test = torchvision.datasets.MNIST('./MNIST_data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)

class Flatten(nn.Module):
    def forward(self, input):
        flattened = input.view(input.shape[0], -1)
        return flattened
    
class Unflatten(nn.Module):
    def __init__(self, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.C = C
        self.H = H
        self.W = W
        
    def forward(self, input):
        unflattened = input.view(-1, self.C, self.H, self.W)
        return unflattened
    
def generate_nosie(batch_size, dim=100):
    # noise = torch.rand(batch_size, dim) * 2 - 1
    noise = torch.rand(batch_size, dim, 1, 1)
    return noise

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1_input = nn.Sequential(
            nn.Conv2d(1, 32, [4,4], stride=[2,2]),
            nn.LeakyReLU(negative_slope=.2)#,
            # nn.MaxPool2d([2,2], stride=[2,2])
        )
        self.layer1_labels = nn.Sequential(
            nn.Conv2d(10, 32, [4,4], stride=[2,2]),
            nn.LeakyReLU(negative_slope=.2)#,
            # nn.MaxPool2d([2,2], stride=[2,2])
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, [4,4], stride=[2,2]),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=.2)
            # nn.MaxPool2d([2,2], stride=[2,2])
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, [4,4], stride=[1,1]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=.2))

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(256, 1, [2,2], stride=[2,2]),
        #     torch.nn.Sigmoid())

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 128, [2,2], stride=[2,2]),
            nn.BatchNorm2d(128))

        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(64,1),
            nn.Sigmoid())

    def forward(self, input, labels):
        # print("input", input.shape)
        # print("labels", labels.shape)
        batch_size = input.shape[0]
        x = self.layer1_input(input)
        y = self.layer1_labels(labels)
        # print("x", x.shape)
        # print("y", y.shape)
        out = torch.cat([x, y], 1)
        # print("out1", out.shape)
        out = self.layer2(out)
        # print("out2", out.shape)
        out = self.layer3(out)
        # print("out3", out.shape)
        out = self.layer4(out)
        # print("out4", out.shape)
        out = out.view(batch_size, -1)
        # print("flattened", out.shape)
        out = self.fc1(out)
        # print("fc1", out.shape)
        out = self.fc2(out)
        # print("fc2", out.shape)
        return out

# class Generator(nn.Module):
#     def __init__(self, noise_dim=96):
#         super(Generator, self).__init__()
#         self.noise_dim = noise_dim
#         self.layer1_input = nn.ConvTranspose2d(96, 256, )
#     def forward(self, input, labels):
#         x = 


class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1_input = nn.Sequential(
            nn.ConvTranspose2d(100, 256, [4,4], stride=[1,1]),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer1_labels = nn.Sequential(
            nn.ConvTranspose2d(10, 256, [4,4], stride=[1,1]),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, [4,4], stride=[2,2]),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, [4,4], stride=[2,2]),
            nn.BatchNorm2d(128))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, [7,7], stride=[1,1]),
            nn.Tanh())

    def forward(self, input, labels):
        # print("input", input.shape)
        # print("label", labels.shape)
        x = self.layer1_input(input)
        y = self.layer1_labels(labels)
        # print("x", x.shape)
        # print("y", y.shape)
        out = torch.cat([x, y], 1)
        # print("out1", out.shape)
        out = self.layer2(out)
        # print("out2", out.shape)
        out = self.layer3(out)
        # print("out3", out.shape)
        out = self.layer4(out)
        # print("out4", out.shape)
        return out

def create_optimizer(model, lr=.01, betas=None):
    if betas == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    return optimizer

def discriminator_loss(scores_real, scores_fake):
    true_labels = torch.ones_like(scores_real)
    valid_loss = torch.mean((scores_real - true_labels) ** 2) * .5
    invalid_loss = torch.mean(scores_fake ** 2) * .5
    loss = valid_loss + invalid_loss
    return loss

def generator_loss(scores_fake):
    true_labels = torch.ones_like(scores_fake)
    loss = torch.mean((scores_fake - true_labels) ** 2) * .5
    return loss

def show_image(images):
#     for image in images:
    images_np = images.detach().numpy().squeeze()
    plt.imshow(images_np[0])
    plt.show()

def save_images(images, epoch, i):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=.05, hspace=.05)

    images = images.data.data.cpu().numpy()[:16]
    for img_num, sample in enumerate(images):
        ax = plt.subplot(gs[img_num])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    filename = "test-" + str(epoch) + "-" + str(i) 
    # print("file logged")
    plt.savefig("./generated_images/" + filename, bbox_inches="tight" )
    plt.close(fig)

    
def train_gan(generator, discriminator, image_loader, epochs, num_train_batches=-1, lr=0.0002):
    generator_optimizer = create_optimizer(generator, lr=lr, betas=(.5, .999))
    discriminator_optimizer = create_optimizer(discriminator, lr=lr, betas=(.5, .999))
    BCE_loss = nn.BCELoss()
    iters = 0
    onehot = torch.zeros(10, 10)
    if use_cuda:
        onehot = onehot.scatter_(1, torch.cuda.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
    else:
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
    fill = torch.zeros([10, 10, 28, 28])
    for i in range(10):
        fill[i, i, :, :] = 1
    for epoch in range(epochs):
        if (epoch+1) == 11:
            #IS ONLY [0] VALID
            generator_optimizer.param_groups[0]["lr"] /= 10 
            discriminator_optimizer.param_groups[0]["lr"] /= 10

        if (epoch+1) == 16:
            generator_optimizer.param_groups[0]["lr"] /= 10
            discriminator_optimizer.param_groups[0]["lr"] /= 10

        for i, (examples, labels) in enumerate(image_loader):
            if use_cuda:
                examples = examples.cuda()
            if i == num_train_batches:
                break
            if examples.shape[0] != batch_size:
                continue

            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            y_fill = fill[labels]
            # # print("y fill", y_fill.shape)
            # print("examples", examples.shape)
            d_logits = discriminator(examples, y_fill).squeeze()
            # print("d_logits", d_logits.shape)
            d_real_loss = BCE_loss(d_logits, torch.ones(batch_size))

            z = generate_nosie(batch_size)
            y_rand = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze()
            # print("y_rand", y_rand.shape)
            y_label = onehot[y_rand]
            y_fill = fill[y_rand]
            # # print("y ffill", y_fill[:,:,0,0].shape)
            images_fake = generator(z, y_label)
            d_result = discriminator(images_fake, y_fill).squeeze()
            d_fake_loss = BCE_loss(d_result, torch.zeros(batch_size))
            d_fake_loss_avg = d_result.data.mean()
            # print("d real loss", d_real_loss)
            # print("d fake loss", d_fake_loss)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            g_loss = BCE_loss(d_result, torch.ones(batch_size))
            g_loss.backward(retain_graph=True)
            generator_optimizer.step()

            if iters % 1000  == 0:
                print("Iteration:", iters)
                print("Epoch:", epoch)
                print("Discriminator Cost", d_loss.cpu().detach().numpy())
                print("Generator Cost", g_loss.cpu().detach().numpy())
                save_images(images_fake, epoch, iters)
            iters += 1                

    return generator, discriminator

generator = Generator()
discriminator = Discriminator()
image_loader = train_loader
epochs = 25
num_train_batches = -1
train_gan(generator, discriminator, image_loader, epochs, num_train_batches=num_train_batches)
