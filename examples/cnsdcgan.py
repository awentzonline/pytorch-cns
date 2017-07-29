from __future__ import print_function
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from cnslib.agent import Agent
from cnslib.argtypes import list_of
from cnslib.genepool import GenePool
from cnslib.genome import ModelGenome


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--save-every', type=int, default=1, help='probability of saving samples')
parser.add_argument('--episode-batches', type=int, default=1)
parser.add_argument('--gene-weight-ratio', type=float, default=0.01)
parser.add_argument('--freq-weight-ratio', type=float, default=0.3)
parser.add_argument('--v-sigma', type=list_of(float), default=1.)
parser.add_argument('--i-sigma', type=float, default=2.)
parser.add_argument('--v-init', type=list_of(float), default=(-1., 1.))
parser.add_argument('--min-genepool', type=int, default=2)
parser.add_argument('--clear-store', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--num-best', type=int, default=20)
parser.add_argument('--num-hidden', type=int, default=256)
parser.add_argument('--model', default='cnn')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
    nc = 1
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
image_shape = (nc, opt.imageSize, opt.imageSize)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class NetG(nn.Module):
    def __init__(self, ngpu):
        super(NetG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        batch, z = input.size()
        input = input.view(batch, z, 1, 1)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class NetD(nn.Module):
    def __init__(self, ngpu):
        super(NetD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)


class MLPD(nn.Module):
    def __init__(self, input_shape, num_hidden):
        super(MLPD, self).__init__()
        num_input = int(np.prod(input_shape))
        print(num_input, num_hidden)
        self.main = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.main(x)
        return y


class MLPG(nn.Module):
    def __init__(self, num_input, num_hidden, output_shape):
        super(MLPG, self).__init__()
        num_output = int(np.prod(output_shape))
        self.output_shape = output_shape
        self.main = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_output),
            nn.Tanh()
        )

    def forward(self, x):
        print(x.size())
        y = self.main(x)
        return y.view(y.size(0), *self.output_shape)


criterion = nn.BCELoss()

noise_shape = (nz,)
input = torch.FloatTensor(opt.batchSize, *image_shape)
noise = torch.FloatTensor(opt.batchSize, *noise_shape)
fixed_noise = torch.FloatTensor(opt.batchSize, *noise_shape).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

def make_mlp_g():
    return MLPG(nz, opt.num_hidden, image_shape)

def make_mlp_d():
    return MLPD(image_shape, opt.num_hidden)

def make_cnn_g():
    netG = NetG(ngpu)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)
    return netG

def make_cnn_d():
    netD = NetD(ngpu)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)
    return netD

if opt.model == 'cnn':
    make_model_g = make_cnn_g
    make_model_d = make_cnn_d
else:
    make_model_g = make_mlp_g
    make_model_d = make_mlp_d

if opt.cuda:
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

def main(config):
    agent_d = Agent(make_model_d(), cuda=config.cuda)
    agent_g = Agent(make_model_g(), cuda=config.cuda)
    best_agent_d = Agent(make_model_d(), cuda=config.cuda)
    best_agent_g = Agent(make_model_g(), cuda=config.cuda)
    for agent in (agent_d, agent_g, best_agent_d, best_agent_g):
        agent.randomize(config.gene_weight_ratio, config.freq_weight_ratio, config.v_init)
        agent.update_model()
    genepool_d = GenePool(key='d_genes')
    genepool_g = GenePool(key='g_genes')
    if config.clear_store:
        genepool_d.clear()
        genepool_g.clear()
    num_episodes = 0
    while True:
        print('Starting discriminator episode')
        reward = run_discriminator_episode(agent_d, best_agent_g, dataloader, config)
        print('Reward {}'.format(reward,))
        best_genomes_d = update_agent(agent_d, reward, genepool_d, config)
        if best_genomes_d:  # make sure the best is still the best
            best_agent_d.load_genome(random.choice(best_genomes_d)[0])
            best_agent_d.update_model()
            reward = run_discriminator_episode(best_agent_d, best_agent_g, dataloader, config)
            genepool_d.report_score(best_agent_d.genome, reward)
        print('Starting generator episode')
        reward = run_generator_episode(best_agent_d, agent_g, dataloader, config)
        print('Reward {}'.format(reward,))
        best_genomes_g = update_agent(agent_g, reward, genepool_g, config)
        if best_genomes_g:
            best_agent_g.load_genome(random.choice(best_genomes_g)[0])
            best_agent_g.update_model()
            reward = run_generator_episode(best_agent_d, best_agent_g, dataloader, config)
            genepool_g.report_score(best_agent_g.genome, reward)
        num_episodes += 1
        if num_episodes % config.save_every == 0 and config.render:
            # vutils.save_image(real_cpu,
            #         '{}/real_samples.png'.format(opt.outf),
            #         normalize=True)
            print('saving')
            fake = best_agent_g(fixed_noise)
            vutils.save_image(fake.data,
                    '{}/fake_samples_epoch_.png'.format(opt.outf, ),
                    normalize=True)


def update_agent(agent, reward, genepool, config):
    best_genomes = genepool.top_n(config.num_best, reverse=False)
    if len(best_genomes) < config.min_genepool:
        genepool.report_score(agent.genome, reward)  # we're still gathering scores
    else:
        _, best_score = best_genomes[0]
        _, worst_best_score = best_genomes[-1]
        print('Genepool top: {}, {}'.format(best_score, worst_best_score))
        if reward > worst_best_score:
            # Our score isn't notable
            agent.crossover(best_genomes)
        else:
            # New low-ish score
            print('new ok score')
            genepool.report_score(agent.genome, reward)
    agent.mutate(index_sigma=config.i_sigma, value_sigma=config.v_sigma)
    agent.update_model()
    return best_genomes


def run_discriminator_episode(agent_d, agent_g, dataloader, config):
    global noisev, fixed_noise  # mutating from pytorch example
    data_iter = iter(dataloader)
    losses = []
    for i in range(config.episode_batches):
        data = next(data_iter)
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if config.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)
        losses.append(criterion(agent_d(inputv), labelv).data[0])
        # train with fake
        noise.resize_(batch_size, *noise_shape).normal_(0, 1)
        noisev = Variable(noise)
        fake = agent_g(noisev)
        labelv = Variable(label.fill_(fake_label))
        losses.append(criterion(agent_d(fake), labelv).data[0])
    return np.sum(losses)


def run_generator_episode(agent_d, agent_g, dataloader, config):
    global noisev, fixed_noise  # mutating from pytorch example
    data_iter = iter(dataloader)
    losses = []
    for i in range(config.episode_batches):
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        losses.append(criterion(agent_d(agent_g(noisev)), labelv).data[0])
        print('[{}/{}][{}/{}]: {} {} / G: {} {}'.format(
            '?', opt.niter, i, len(dataloader),
            np.min(losses), np.max(losses), np.min(losses), np.max(losses)))
    return np.sum(losses)


if __name__ == '__main__':
    main(opt)
