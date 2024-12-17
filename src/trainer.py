import sys
sys.path.insert(0, "../model")

import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from txt2image_dataset import Text2ImageDataset    #txt2image_dataset.ipynb file
from gan_factory import gan_factory
from utils import Utils, Logger    #utils.ipynb file
from PIL import Image
import os

class Trainer(object):
    def __init__(self, type, dataset, split, lr, save_path, l1_coef, l2_coef, pre_trained_gen, pre_trained_disc, batch_size, num_workers, epochs, device):
        with open('/Users/userd/Desktop/final_ml_project/config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.device = device  # Save the device
        print(f"Trainer initialized with device: {self.device}")

        self.generator = torch.nn.DataParallel(gan_factory.generator_factory(type).to(self.device))
        self.discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory(type).to(self.device))

        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc, map_location=self.device))
        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen, map_location=self.device))
        else:
            self.generator.apply(Utils.weights_init)

        if dataset == 'birds':
            self.dataset = Text2ImageDataset(config['birds_dataset_path'], split=split)
        elif dataset == 'flowers':
            self.dataset = Text2ImageDataset(config['flowers_dataset_path'], split=split)
        else:
            print('Dataset not supported, please select either birds or flowers.')
            exit()

        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.logger = Logger()
        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path
        self.type = type

    def train(self, cls):

        if self.type == 'gan':
            self._train_gan(cls)


    def _train_gan(self, cls):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        self.d_losses = []  # Discriminator loss
        self.g_losses = []  # Generator loss
        self.d_x = []       # D(X): Real image confidence
        self.d_gx = []      # D(G(X)): Fake image confidence


        for epoch in range(self.num_epochs):
            iteration = 0
            #print 'Outside Dataloader loop is'
            for sample in self.data_loader:
                #print 'Inside Dataloader loop is'
                iteration += 1
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                wrong_images = sample['wrong_images']

                right_images = Variable(right_images.float()).to(self.device)
                right_embed = Variable(right_embed.float()).to(self.device)
                wrong_images = Variable(wrong_images.float()).to(self.device)
                
                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).to(self.device)
                smoothed_real_labels = Variable(smoothed_real_labels).to(self.device)
                fake_labels = Variable(fake_labels).to(self.device)

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs, _ = self.discriminator(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                # noise = Variable(torch.randn(right_images.size(0), 100)).to(self.device)
                # noise = noise.view(noise.size(0), 100, 1, 1)
                # Noise tensor for the generator
                noise = Variable(torch.randn(right_images.size(0), 100)).to(self.device)
                noise = noise.view(noise.size(0), 100, 1, 1)

                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss

                if cls:
                    d_loss = d_loss + wrong_loss

                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).to(self.device)
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)
                _, activation_real = self.discriminator(right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)    #try with median and check if it converges
                activation_real = torch.mean(activation_real, 0)    #try with median and check if it converges


                g_loss = criterion(outputs, real_labels)                          + self.l2_coef * l2_loss(activation_fake, activation_real.detach())                          + self.l1_coef * l1_loss(fake_images, right_images)

                g_loss.backward()
                self.optimG.step()
                
#                print('iter:', iteration)

                # if iteration % 5 == 0:
                #     self.logger.log_iteration_gan(epoch, iteration, d_loss, g_loss, real_score, fake_score)
                if iteration % 5 == 0:
                    self.d_losses.append(d_loss.item())
                    self.g_losses.append(g_loss.item())
                    self.d_x.append(real_score.mean().item())
                    self.d_gx.append(fake_score.mean().item())

                    self.logger.log_iteration_gan(epoch, iteration, d_loss, g_loss, real_score, fake_score)


            # Save training logs after training ends
            np.savez('training_logs.npz', 
                    d_losses=self.d_losses, 
                    g_losses=self.g_losses, 
                    d_x=self.d_x, 
                    d_gx=self.d_gx)
            print("Training logs saved to 'training_logs.npz'")

            if (epoch) % 10 == 0:
                #print('epoch', epoch, 'complete')
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)


    def predict(self):
        for sample in self.data_loader:
            #print len(sample)
            right_images = sample['right_images']
            right_embed = sample['right_embed']
            #print type(right_embed)
            #print right_embed.shape
            #break
            txt = sample['txt']

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            right_images = Variable(right_images.float()).to(self.device)
            right_embed = Variable(right_embed.float()).to(self.device)

            # Train the generator
            noise = Variable(torch.randn(right_images.size(0), 100)).to(self.device)
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = self.generator(right_embed, noise)


            for image, t in zip(fake_images, txt):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))
                print(t)
            break

