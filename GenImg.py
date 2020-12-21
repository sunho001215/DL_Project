import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import cv2
import numpy as np
import argparse
from model import *
from VisT import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate Image")
    
    parser.add_argument('--train_case', default= 5, help='choose your train case', type=int)
    parser.add_argument('--num_image', default= 2048, help='choose number of image to generate',type=int)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train_case == 1:
        G = Generator(3, 100, 64).to(device)
        G.load_state_dict(torch.load('/home/sunho/DL_Project/CelebA_DCGAN_results/60_epochs/Case1/backup/generator_param_59.pkl'))
        G.eval()

        num = 0
        for i in range(int(args.num_image/512)):
            z_ = torch.randn((512, 100)).view(-1, 100, 1, 1)
            with torch.no_grad():
                z_ = Variable(z_.cuda())
            images = G(z_)

            for k in range(512):
                image = (images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) * 255/2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite("/home/sunho/DL_Project/result_images/Case1/image_%d.jpg" % (num), image)
                num+=1

    if args.train_case == 2:
        G = GenWithConv(3, 100, 64).to(device)
        G.load_state_dict(torch.load('/home/sunho/DL_Project/CelebA_DCGAN_results/60_epochs/Case2/backup/generator_param_59.pkl'))
        G.eval()

        num = 0
        for i in range(int(args.num_image/512)):
            z_ = torch.randn((512, 100)).view(-1, 100, 1, 1)
            with torch.no_grad():
                z_ = Variable(z_.cuda())
            _, images = G(z_)

            for k in range(512):
                image = (images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) * 255/2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite("/home/sunho/DL_Project/result_images/Case2/image_%d.jpg" % (num), image)
                num+=1

    if args.train_case == 3:
        G = VisTGen().to(device)
        G.load_state_dict(torch.load('/home/sunho/DL_Project/CelebA_DCGAN_results/60_epochs/Case3/backup/generator_param_59.pkl'))
        G.eval()

        num = 0
        for i in range(int(args.num_image/512)):
            z_ = torch.randn((512, 100)).view(-1, 100, 1, 1)
            with torch.no_grad():
                z_ = Variable(z_.cuda())
            _, images = G(z_)

            for k in range(512):
                image = (images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) * 255/2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite("/home/sunho/DL_Project/result_images/Case3/image_%d.jpg" % (num), image)
                num+=1

    if args.train_case == 4:
        G = GenWithConv(3, 100, 64).to(device)
        G.load_state_dict(torch.load('/home/sunho/DL_Project/CelebA_DCGAN_results/60_epochs/Case4/backup/generator_param_59.pkl'))
        G.eval()

        num = 0
        for i in range(int(args.num_image/512)):
            z_ = torch.randn((512, 100)).view(-1, 100, 1, 1)
            with torch.no_grad():
                z_ = Variable(z_.cuda())
            _, images = G(z_)

            for k in range(512):
                image = (images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) * 255/2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite("/home/sunho/DL_Project/result_images/Case4/image_%d.jpg" % (num), image)
                num+=1

    if args.train_case == 5:
        G = Generator(3, 100, 64).to(device)
        G.load_state_dict(torch.load('/home/sunho/DL_Project/CelebA_DCGAN_results/60_epochs/Case5/backup/generator_param_54.pkl'))
        G.eval()

        num = 0
        for i in range(int(args.num_image/512)):
            z_ = torch.randn((512, 100)).view(-1, 100, 1, 1)
            with torch.no_grad():
                z_ = Variable(z_.cuda())
            images = G(z_)

            for k in range(512):
                image = (images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) * 255/2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite("/home/sunho/DL_Project/result_images/Case5/image_%d.jpg" % (num), image)
                num+=1



        