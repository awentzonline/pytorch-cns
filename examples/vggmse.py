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
from torchvision.models.vgg import vgg16

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
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--sigma', type=float, default=0.1, help='default=0.1')
parser.add_argument('--population-size', type=int, default=5)
parser.add_argument('--save-every', type=float, default=0.1, help='probability of saving samples')
parser.add_argument('--episode-batches', type=int, default=5)
parser.add_argument('--max-genes', type=int, default=20)
parser.add_argument('--min-genes', type=int, default=10)
parser.add_argument('--v-change', type=list_of(float), default=(-1., 1.))
parser.add_argument('--v-init', type=list_of(float), default=(-10., 10.))
parser.add_argument('--min-genepool', type=int, default=10)
parser.add_argument('--clear-store', action='store_true')
parser.add_argument('--render', action='store_true')
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
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

netG = NetG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

criterion = nn.MSELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
vgg = vgg16(pretrained=True)
vgg_features = nn.Sequential(*(vgg.features[i] for i in range(11)))
if opt.cuda:
    criterion.cuda()
    input, noise, fixed_noise = input.cuda(), noise.cuda(), fixed_noise.cuda()
    vgg_features.cuda()

fixed_noise = Variable(fixed_noise)

def main(config):
    agent_g = Agent(netG)
    agent_g.randomize(config.min_genes, config.max_genes, config.v_init)
    genepool_g = GenePool(key='g_genes_vggmse')
    if config.clear_store:
        genepool_g.clear()
    num_episodes = 0
    while True:
        print('Starting generator episode')
        reward = run_generator_episode(agent_g, vgg_features, dataloader, config)
        print('Reward {}'.format(reward,))
        update_agent(agent_g, reward, genepool_g, config)
        num_episodes += 1


def update_agent(agent, reward, genepool, config):
    best_genomes = genepool.top_n(config.min_genepool, reverse=False)
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
    agent.mutate()
    agent.update_model()


def run_generator_episode(agent_g, vgg_features, dataloader, config):
    global noisev, fixed_noise  # mutating from pytorch example
    data_iter = iter(dataloader)
    losses = []
    for i in range(config.episode_batches):
        data = next(data_iter)
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if config.cuda:
            real_cpu = real_cpu.cuda()
        # real
        input.resize_as_(real_cpu).copy_(real_cpu)
        inputv = Variable(input)
        real_features = vgg_features(inputv).detach()
        # fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = agent_g(noisev).detach()
        fake_features = vgg_features(fake)
        losses.append(criterion(fake_features, real_features).data[0])

        print('[{}/{}][{}/{}]: {} {} / G: {} {}'.format(
            '?', opt.niter, i, len(dataloader),
            np.min(losses), np.max(losses), np.min(losses), np.max(losses)))
        if config.save_every < np.random.uniform() and config.render:
            # vutils.save_image(real_cpu,
            #         '{}/real_samples.png'.format(opt.outf),
            #         normalize=True)
            fake = agent_g(fixed_noise)
            vutils.save_image(fake.data,
                    '{}/fake_samples_epoch_.png'.format(opt.outf,),
                    normalize=True)
    return np.sum(losses)
#

if __name__ == '__main__':
    main(opt)
