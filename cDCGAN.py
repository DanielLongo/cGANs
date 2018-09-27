import torch
from torch import nn
import torchvision.datasets
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
    
def generate_nosie(batch_size, dim=100):
    noise = torch.randn(batch_size, dim, 1, 1)
    return noise

class DiscrimanatorOrig(nn.Module):
    def __init__(self):
        super(DiscrimanatorOrig, self).__init__()
        self.layer1_input = nn.Sequential(
            nn.Conv2d(1, 64, [4,4], stride=[2,2]),
            nn.LeakyReLU(negative_slope=.2)
        )
        self.layer1_labels = nn.Sequential(
            nn.Conv2d(10, 64, [4,4], stride=[2,2]),
            nn.LeakyReLU(negative_slope=.2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, [4,4], stride=[2,2]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, [4,4], stride=[2,2]),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=.2)
        )
        self.layer4 = nn.Sequential (
            nn.Conv2d(512, 1, [2,2], stride=[1,1]),
            nn.Sigmoid())

    def forward(self, input, labels):
        x = self.layer1_input(input)
        y = self.layer1_labels(labels)
        out = torch.cat([x,y], 1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def weight_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1_input = nn.Sequential(
            nn.Conv2d(1, 64, [4,4], stride=[2,2]),
            nn.LeakyReLU(negative_slope=.2)
        )
        self.layer1_labels = nn.Sequential(
            nn.Conv2d(10, 64, [4,4], stride=[2,2]),
            nn.LeakyReLU(negative_slope=.2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, [4,4], stride=[2,2]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, [4,4], stride=[1,1]),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=.2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 256, [2,2], stride=[2,2]),
            nn.BatchNorm2d(256))

        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid())

    def weight_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def forward(self, input, labels):
        batch_size = input.shape[0]
        x = self.layer1_input(input)
        y = self.layer1_labels(labels)
        out = torch.cat([x, y], 1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(batch_size, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class ConditionalGenerator(nn.Module):
    # initializers
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        self.layer1_input = nn.Sequential(
            nn.ConvTranspose2d(100, 256, [4,4], stride=[1,1]),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer1_labels = nn.Sequential(
            nn.ConvTranspose2d(10, 256, [4,4], stride=[1,1]),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, [4,4], stride=[2,2], padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, [4,4], stride=[2,2], padding=1),
            nn.BatchNorm2d(128))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, [4,4], stride=[2,2], padding=1),
            nn.Tanh())

    def forward(self, input, labels):
        x = self.layer1_input(input)
        y = self.layer1_labels(labels)
        out = torch.cat([x, y], 1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
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

def save_images(generator, epoch, i):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=.05, hspace=.05)
    z = generate_nosie(100)
    if use_cuda:
        onehot = torch.zeros(10, 10).scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda().view(10,1), 1).view(10, 10, 1, 1)
        fill = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10).cuda()
    else:
        onehot = torch.zeros(10, 10).scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
        fill = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10)
    fill = onehot[fill]

    images_fake = generator(z, fill)
    images_fake = images_fake.data.data.cpu().numpy()
    for img_num, sample in enumerate(images_fake):
        ax = plt.subplot(gs[img_num])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(32, 32), cmap='Greys_r')

    filename = "test-" + str(epoch) + "-" + str(i) 
    plt.savefig("./generated_images_EMNIST/" + filename, bbox_inches="tight" )
    plt.close(fig)

    
def train_gan(generator, discriminator, image_loader, epochs, num_train_batches=-1, lr=0.0002):
    generator_optimizer = create_optimizer(generator, lr=lr, betas=(.5, .999))
    discriminator_optimizer = create_optimizer(discriminator, lr=lr, betas=(.5, .999))
    BCE_loss = nn.BCELoss()
    iters = 0
    onehot = torch.zeros(10, 10)
    if use_cuda:
        onehot = onehot.scatter_(1, torch.cuda.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).type(dtype).view(10, 10, 1, 1)
    else:
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).type(dtype).view(10, 10, 1, 1)

    fill = torch.zeros([10, 10, 32, 32])
    for i in range(10):
        fill[i, i, :, :] = 1
    for epoch in range(epochs):
        if ((epoch+1) == 11) or ((epoch+1) == 16):
            generator_optimizer.param_groups[0]["lr"] /= 10 
            discriminator_optimizer.param_groups[0]["lr"] /= 10

        for i, (examples, labels) in enumerate(image_loader):
            examples = examples.type(dtype)
            if i == num_train_batches:
                break
            if examples.shape[0] != batch_size:
                continue

            discriminator_optimizer.zero_grad()

            y_fill = fill[labels]
            d_logits = discriminator(examples, y_fill).squeeze()
            d_real_loss = BCE_loss(d_logits, torch.ones(batch_size))
            z = generate_nosie(batch_size)

            if use_cuda:
                y_rand = (torch.rand(batch_size, 1) * 10).type(torch.cuda.LongTensor).squeeze()            
            else:
                y_rand = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze()

            y_label = onehot[y_rand]
            y_fill = fill[y_rand]
            images_fake = generator(z, y_label)
            d_result = discriminator(images_fake, y_fill).squeeze()
            d_fake_loss = BCE_loss(d_result, torch.zeros(batch_size))
            d_fake_loss_avg = d_result.data.mean()
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            discriminator_optimizer.step()

            # train generator seperatly
            generator_optimizer.zero_grad()
            z = generate_nosie(batch_size)
            if use_cuda:
                y_rand = (torch.rand(batch_size, 1) * 10).type(torch.cuda.LongTensor).squeeze()            
            else:
                y_rand = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze()
            y_label = onehot[y_rand]
            y_fill = fill[y_rand]
            images_fake = generator(z, y_label)
            d_result = discriminator(images_fake, y_fill).squeeze()
            g_loss = BCE_loss(d_result, torch.ones(batch_size))
            g_loss.backward(retain_graph=True)
            g_loss.backward()
            generator_optimizer.step()
            iters += 1

        print("Iteration:", iters)
        print("Epoch:", epoch)
        print("Discriminator Cost", d_loss.cpu().detach().numpy())
        print("Generator Cost", g_loss.cpu().detach().numpy())
        save_images(generator, epoch, iters)

    return generator, discriminator

if __name__ == "__main__":
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

    # mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
    mnist_train = torchvision.datasets.EMNIST('./EMNIST_data', train=True, download=True, transform=transform, split="letters")
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test = torchvision.datasets.MNIST('./MNIST_data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,  shuffle=True) 
    
    generator = ConditionalGenerator()
    generator.weight_init(mean=0.0, std=0.02)
    discriminator = DiscrimanatorOrig()
    discriminator.weight_init(mean=0.0, std=0.02)
    image_loader = train_loader
    epochs = 20
    num_train_batches = -1
    generator, discriminator = train_gan(generator, discriminator, image_loader, epochs, num_train_batches=num_train_batches)
    print("Training finished")
    # torch.save(generator, generator_filename + ".pt")
    # torch.save(discriminator, discriminator_filename + ".pt")
    torch.save(generator.state_dict(), generator_filename + ".pt")
    torch.save(discriminator.state_dict(), discriminator_filename + ".pt")
    print("Models Saved")
