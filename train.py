"""
https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN

Based on the code above
"""

from tqdm import tqdm
import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
from model import *
from VisT import *

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
with torch.no_grad():
    fixed_z_ = Variable(fixed_z_.cuda())

def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    with torch.no_grad():
        z_ = Variable(z_.cuda())

    if args.train_case == 3 or args.train_case == 4:
        G.eval()
        if isFix:
            _, test_images = G(fixed_z_)
        else:
            _, test_images = G(z_)
        G.train()
    
    else:
        G.eval()
        if isFix:
            test_images = G(fixed_z_)
        else:
            test_images = G(z_)
        G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training Code for Deep Learning Project")
    
    parser.add_argument('--batch_size', default= 512, help='batch size', type=int)
    parser.add_argument('--learning_rate', default= 0.0002, help='learning rate', type=float)
    parser.add_argument('--train_epochs', default= 20, help='train epochs', type=int)
    parser.add_argument('--data_dir', default= 'data/resized_celebA', help='train data directory')
    # 1 : train original DCGAN
    # 2 : train DCGAN + additional conv layer
    # 3 : train DCGAN + visual transformer
    # 4 : train DCGAN + additional conv layer with original DCGAN backbone
    # 5 : train DCGAN + visual transformer with original DCGAN backbone
    parser.add_argument('--train_case', default= 1, help='choose your train case', type=int)

    args = parser.parse_args()

    img_size = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]) 
    
    #print(args.data_dir)
    data_set = datasets.ImageFolder(args.data_dir, transform)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size = args.batch_size, shuffle = True)

    data_check = plt.imread(train_loader.dataset.imgs[0][0])
    if (data_check.shape[0] != img_size) or (data_check.shape[1] != img_size):
        sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.ipynb\" !!!')
        sys.exit(1)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train_case == 1:
        print("Train Case : 1")
        G = Generator(3, 100, 64).to(device)
        D = Discriminator(3, 64).to(device)
        #G.weight_init(mean = 0.0, std = 0.02)
        #D.weight_init(mean = 0.0, std = 0.02)
        weights_init(G)
        weights_init(D)
        #G.cuda()
        #D.cuda()

        BCE_loss =nn.BCELoss()

        #print(args.learning_rate)
        G_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        if not os.path.isdir('CelebA_DCGAN_results'):
            os.mkdir('CelebA_DCGAN_results')
        if not os.path.isdir('CelebA_DCGAN_results/Case1'):
            os.mkdir('CelebA_DCGAN_results/Case1')
        if not os.path.isdir('CelebA_DCGAN_results/Case1/Random_results'):
            os.mkdir('CelebA_DCGAN_results/Case1/Random_results')
        if not os.path.isdir('CelebA_DCGAN_results/Case1/Fixed_results'):
            os.mkdir('CelebA_DCGAN_results/Case1/Fixed_results')
        
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []

        print('Training start!')
        start_time = time.time()
        G.train()
        D.train()
        for epoch in range(args.train_epochs):
            D_losses = []
            G_losses = []

            num_iter = 0

            epoch_start_time = time.time()
            for x_, _ in train_loader:

                # train discriminator D
                D.zero_grad()

                mini_batch = x_.size()[0]

                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)

                x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
                D_result = D(x_).squeeze()
                D_real_loss = BCE_loss(D_result, y_real_)

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                z_ = Variable(z_.cuda())
                G_result = G(z_)

                D_result = D(G_result).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_)
                D_fake_score = D_result.data.mean()

                D_train_loss = D_real_loss + D_fake_loss
                #print(D_train_loss)

                D_train_loss.backward()
                D_optimizer.step()

                D_losses.append(D_train_loss.item())

                # train generator G
                G.zero_grad()

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                z_ = Variable(z_.cuda())

                G_result = G(z_)
                D_result = D(G_result).squeeze()
                G_train_loss = BCE_loss(D_result, y_real_)
                G_train_loss.backward()
                #print(G_train_loss)
                G_optimizer.step()

                G_losses.append(G_train_loss.item())

                num_iter += 1
                #if num_iter % 50 == 0:
                #    print("Training...")
                if num_iter % 10 == 0:
                    print("[num_iter : %d] D_train_loss : %f, G_train_loss : %f" % (num_iter, D_train_loss.item(), G_train_loss.item()))

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time


            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), args.train_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                    torch.mean(torch.FloatTensor(G_losses))))
            p = 'CelebA_DCGAN_results/Case1/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
            fixed_p = 'CelebA_DCGAN_results/Case1/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
            show_result((epoch+1), save=True, path=p, isFix=False)
            show_result((epoch+1), save=True, path=fixed_p, isFix=True)
            train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
            train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

            if (epoch+1) % 5 ==0:
                torch.save(G.state_dict(), "CelebA_DCGAN_results/Case1/generator_param_%d.pkl" % (epoch))
                torch.save(D.state_dict(), "CelebA_DCGAN_results/Case1/discriminator_param_%d.pkl" % (epoch))

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), args.train_epochs, total_ptime))
        print("Training finish!... save training results")
        #torch.save(G.state_dict(), "CelebA_DCGAN_results/Case1/generator_param.pkl")
        #torch.save(D.state_dict(), "CelebA_DCGAN_results/Case1/discriminator_param.pkl")
        with open('CelebA_DCGAN_results/Case1/train_hist.pkl', 'wb') as f:
            pickle.dump(train_hist, f)

        show_train_hist(train_hist, save=True, path='CelebA_DCGAN_results/Case1/CelebA_DCGAN_train_hist.png')

        images = []
        for e in range(args.train_epochs):
            img_name = 'CelebA_DCGAN_results/Case1/Fixed_results/CelebA_DCGAN_' + str(e + 1) + '.png'
            images.append(imageio.imread(img_name))
        imageio.mimsave('CelebA_DCGAN_results/Case1/generation_animation.gif', images, fps=5)

    if args.train_case == 2:
        print("Train Case : 2")
        G = GenWithConv(3, 100, 64).to(device)
        D = Discriminator(3, 64).to(device)
        #G.weight_init(mean = 0.0, std = 0.02)
        #D.weight_init(mean = 0.0, std = 0.02)
        weights_init(D)
        weights_init(G)
        #G.cuda()
        #D.cuda()

        BCE_loss =nn.BCELoss()

        #print(args.learning_rate)
        G_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        if not os.path.isdir('CelebA_DCGAN_results'):
            os.mkdir('CelebA_DCGAN_results')
        if not os.path.isdir('CelebA_DCGAN_results/Case2'):
            os.mkdir('CelebA_DCGAN_results/Case2')
        if not os.path.isdir('CelebA_DCGAN_results/Case2/Random_results'):
            os.mkdir('CelebA_DCGAN_results/Case2/Random_results')
        if not os.path.isdir('CelebA_DCGAN_results/Case2/Fixed_results'):
            os.mkdir('CelebA_DCGAN_results/Case2/Fixed_results')
        
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []

        print('Training start!')
        start_time = time.time()
        G.train()
        D.train()
        for epoch in range(args.train_epochs):
            D_losses = []
            G_losses = []

            num_iter = 0

            epoch_start_time = time.time()
            for x_, _ in train_loader:

                # train discriminator D
                D.zero_grad()

                mini_batch = x_.size()[0]

                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)

                x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
                D_result = D(x_).squeeze()
                D_real_loss = BCE_loss(D_result, y_real_)

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                z_ = Variable(z_.cuda())
                G_result = G(z_)

                D_result = D(G_result).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_)
                D_fake_score = D_result.data.mean()

                D_train_loss = D_real_loss + D_fake_loss
                #print(D_train_loss)

                D_train_loss.backward()
                D_optimizer.step()

                D_losses.append(D_train_loss.item())

                # train generator G
                G.zero_grad()

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                z_ = Variable(z_.cuda())

                G_result = G(z_)
                D_result = D(G_result).squeeze()
                G_train_loss = BCE_loss(D_result, y_real_)
                G_train_loss.backward()
                #print(G_train_loss)
                G_optimizer.step()

                G_losses.append(G_train_loss.item())

                num_iter += 1
                #if num_iter % 50 == 0:
                #    print("Training...")
                if num_iter % 10 == 0:
                    print("[num_iter : %d] D_train_loss : %f, G_train_loss : %f" % (num_iter, D_train_loss.item(), G_train_loss.item()))

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time


            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), args.train_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                    torch.mean(torch.FloatTensor(G_losses))))
            p = 'CelebA_DCGAN_results/Case2/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
            fixed_p = 'CelebA_DCGAN_results/Case2/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
            show_result((epoch+1), save=True, path=p, isFix=False)
            show_result((epoch+1), save=True, path=fixed_p, isFix=True)
            train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
            train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

            if (epoch+1) % 5 ==0:
                torch.save(G.state_dict(), "CelebA_DCGAN_results/Case2/generator_param_%d.pkl" % (epoch))
                torch.save(D.state_dict(), "CelebA_DCGAN_results/Case2/discriminator_param_%d.pkl" % (epoch))

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), args.train_epochs, total_ptime))
        print("Training finish!... save training results")
        #torch.save(G.state_dict(), "CelebA_DCGAN_results/Case2/generator_param.pkl")
        #torch.save(D.state_dict(), "CelebA_DCGAN_results/Case2/discriminator_param.pkl")
        with open('CelebA_DCGAN_results/Case2/train_hist.pkl', 'wb') as f:
            pickle.dump(train_hist, f)

        show_train_hist(train_hist, save=True, path='CelebA_DCGAN_results/Case2/CelebA_DCGAN_train_hist.png')

        images = []
        for e in range(args.train_epochs):
            img_name = 'CelebA_DCGAN_results/Case2/Fixed_results/CelebA_DCGAN_' + str(e + 1) + '.png'
            images.append(imageio.imread(img_name))
        imageio.mimsave('CelebA_DCGAN_results/Case2/generation_animation.gif', images, fps=5)

    if args.train_case == 3:
        print("Train Case : 3")
        G = VisTGen().to(device)
        D = Discriminator(3, 64).to(device)
        #G.weight_init(mean = 0.0, std = 0.02)
        #D.weight_init(mean = 0.0, std = 0.02)
        weights_init(D)
        weights_init(G)
        #G.cuda()
        #D.cuda()

        BCE_loss =nn.BCELoss()

        #print(args.learning_rate)
        G_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        if not os.path.isdir('CelebA_DCGAN_results'):
            os.mkdir('CelebA_DCGAN_results')
        if not os.path.isdir('CelebA_DCGAN_results/Case3'):
            os.mkdir('CelebA_DCGAN_results/Case3')
        if not os.path.isdir('CelebA_DCGAN_results/Case3/Random_results'):
            os.mkdir('CelebA_DCGAN_results/Case3/Random_results')
        if not os.path.isdir('CelebA_DCGAN_results/Case3/Fixed_results'):
            os.mkdir('CelebA_DCGAN_results/Case3/Fixed_results')
        
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []

        print('Training start!')
        start_time = time.time()
        G.train()
        D.train()
        for epoch in range(args.train_epochs):
            D_losses = []
            G_losses = []

            num_iter = 0
            alpha = 0.5 + 0.5 * (epoch + 1)/(args.train_epochs)
            epoch_start_time = time.time()
            for x_, _ in train_loader:

                # train discriminator D
                D.zero_grad()

                mini_batch = x_.size()[0]

                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)

                x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
                D_result = D(x_).squeeze()
                D_real_loss = BCE_loss(D_result, y_real_)

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                z_ = Variable(z_.cuda())
                G_pri, G_result = G(z_)

                D_result = D(G_result).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_)
                #D_fake_score = D_result.data.mean()

                D_result_pri = D(G_pri).squeeze()
                D_fake_loss_pri = BCE_loss(D_result_pri, y_fake_)
                #D_fake_score_pri = D_result_pri.data.mean()

                D_train_loss = D_real_loss + alpha * D_fake_loss + (1-alpha) * D_fake_loss_pri
                #D_train_loss = D_real_loss + D_fake_loss
                #print(D_train_loss)

                D_train_loss.backward()
                D_optimizer.step()

                D_losses.append(D_train_loss.item())

                # train generator G
                G.zero_grad()

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                z_ = Variable(z_.cuda())

                G_pri, G_result = G(z_)

                D_result = D(G_result).squeeze()
                G_train_loss_fin = BCE_loss(D_result, y_real_)

                D_pri_result = D(G_pri).squeeze()
                G_train_loss_pri = BCE_loss(D_pri_result, y_real_)

                G_train_loss = alpha * G_train_loss_fin + (1-alpha) * G_train_loss_pri
                #G_train_loss = G_train_loss_fin

                G_train_loss.backward()
                #print(G_train_loss)
                G_optimizer.step()

                G_losses.append(G_train_loss.item())

                num_iter += 1
                #if num_iter % 50 == 0:
                #    print("Training...")
                if num_iter % 10 == 0:
                    print("[num_iter : %d] D_train_loss : %f, G_train_loss : %f" % (num_iter, D_train_loss.item(), G_train_loss.item()))

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time


            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), args.train_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                    torch.mean(torch.FloatTensor(G_losses))))
            p = 'CelebA_DCGAN_results/Case3/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
            fixed_p = 'CelebA_DCGAN_results/Case3/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
            show_result((epoch+1), save=True, path=p, isFix=False)
            show_result((epoch+1), save=True, path=fixed_p, isFix=True)
            train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
            train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

            if (epoch+1) % 5 ==0:
                torch.save(G.state_dict(), "CelebA_DCGAN_results/Case3/generator_param_%d.pkl" % (epoch))
                torch.save(D.state_dict(), "CelebA_DCGAN_results/Case3/discriminator_param_%d.pkl" % (epoch))

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), args.train_epochs, total_ptime))
        print("Training finish!... save training results")
        #torch.save(G.state_dict(), "CelebA_DCGAN_results/Case3/generator_param.pkl")
        #torch.save(D.state_dict(), "CelebA_DCGAN_results/Case3/discriminator_param.pkl")
        with open('CelebA_DCGAN_results/Case3/train_hist.pkl', 'wb') as f:
            pickle.dump(train_hist, f)

        show_train_hist(train_hist, save=True, path='CelebA_DCGAN_results/Case3/CelebA_DCGAN_train_hist.png')

        images = []
        for e in range(args.train_epochs):
            img_name = 'CelebA_DCGAN_results/Case3/Fixed_results/CelebA_DCGAN_' + str(e + 1) + '.png'
            images.append(imageio.imread(img_name))
        imageio.mimsave('CelebA_DCGAN_results/Case3/generation_animation.gif', images, fps=5)
    
    if args.train_case == 4:
        print("Train Case : 4")
        G = GenWithConv(3, 100, 64).to(device)
        D = Discriminator(3, 64).to(device)
        #G.weight_init(mean = 0.0, std = 0.02)
        #D.weight_init(mean = 0.0, std = 0.02)
        weights_init(G)
        weights_init(D)
        #G.cuda()
        #D.cuda()

        BCE_loss =nn.BCELoss()

        #print(args.learning_rate)
        G_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        if not os.path.isdir('CelebA_DCGAN_results'):
            os.mkdir('CelebA_DCGAN_results')
        if not os.path.isdir('CelebA_DCGAN_results/Case4'):
            os.mkdir('CelebA_DCGAN_results/Case4')
        if not os.path.isdir('CelebA_DCGAN_results/Case4/Random_results'):
            os.mkdir('CelebA_DCGAN_results/Case4/Random_results')
        if not os.path.isdir('CelebA_DCGAN_results/Case4/Fixed_results'):
            os.mkdir('CelebA_DCGAN_results/Case4/Fixed_results')
        
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []

        print('Training start!')
        start_time = time.time()
        G.train()
        D.train()
        for epoch in range(args.train_epochs):
            D_losses = []
            G_losses = []

            num_iter = 0
            alpha = 0.5 + 0.5 * (epoch + 1)/(args.train_epochs)
            epoch_start_time = time.time()
            for x_, _ in train_loader:

                # train discriminator D
                D.zero_grad()

                mini_batch = x_.size()[0]

                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)

                x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
                D_result = D(x_).squeeze()
                D_real_loss = BCE_loss(D_result, y_real_)

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                z_ = Variable(z_.cuda())
                G_pri, G_result = G(z_)

                D_result = D(G_result).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_)
                #D_fake_score = D_result.data.mean()

                D_result_pri = D(G_pri).squeeze()
                D_fake_loss_pri = BCE_loss(D_result_pri, y_fake_)
                #D_fake_score_pri = D_result_pri.data.mean()

                D_train_loss = D_real_loss + alpha * D_fake_loss + (1-alpha) * D_fake_loss_pri
                #D_train_loss = D_real_loss + D_fake_loss
                #print(D_train_loss)

                D_train_loss.backward()
                D_optimizer.step()

                D_losses.append(D_train_loss.item())

                # train generator G
                G.zero_grad()

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                z_ = Variable(z_.cuda())

                G_pri, G_result = G(z_)

                D_result = D(G_result).squeeze()
                G_train_loss_fin = BCE_loss(D_result, y_real_)

                D_pri_result = D(G_pri).squeeze()
                G_train_loss_pri = BCE_loss(D_pri_result, y_real_)

                G_train_loss = alpha * G_train_loss_fin + (1-alpha) * G_train_loss_pri
                #G_train_loss = G_train_loss_fin

                G_train_loss.backward()
                #print(G_train_loss)
                G_optimizer.step()

                G_losses.append(G_train_loss.item())

                num_iter += 1
                #if num_iter % 50 == 0:
                #    print("Training...")
                if num_iter % 10 == 0:
                    print("[num_iter : %d] D_train_loss : %f, G_train_loss : %f" % (num_iter, D_train_loss.item(), G_train_loss.item()))

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), args.train_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                    torch.mean(torch.FloatTensor(G_losses))))
            p = 'CelebA_DCGAN_results/Case4/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
            fixed_p = 'CelebA_DCGAN_results/Case4/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
            show_result((epoch+1), save=True, path=p, isFix=False)
            show_result((epoch+1), save=True, path=fixed_p, isFix=True)
            train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
            train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

            if (epoch+1) % 5 ==0:
                torch.save(G.state_dict(), "CelebA_DCGAN_results/Case4/generator_param_%d.pkl" % (epoch))
                torch.save(D.state_dict(), "CelebA_DCGAN_results/Case4/discriminator_param_%d.pkl" % (epoch))

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), args.train_epochs, total_ptime))
        print("Training finish!... save training results")
        #torch.save(G.state_dict(), "CelebA_DCGAN_results/Case4/generator_param.pkl")
        #torch.save(D.state_dict(), "CelebA_DCGAN_results/Case4/discriminator_param.pkl")
        with open('CelebA_DCGAN_results/Case4/train_hist.pkl', 'wb') as f:
            pickle.dump(train_hist, f)

        show_train_hist(train_hist, save=True, path='CelebA_DCGAN_results/Case4/CelebA_DCGAN_train_hist.png')

        images = []
        for e in range(args.train_epochs):
            img_name = 'CelebA_DCGAN_results/Case4/Fixed_results/CelebA_DCGAN_' + str(e + 1) + '.png'
            images.append(imageio.imread(img_name))
        imageio.mimsave('CelebA_DCGAN_results/Case4/generation_animation.gif', images, fps=5)


    if args.train_case == 5:
        print("Train Case : 5")
        G = Generator(3, 100, 64).to(device)
        D = VisTDis().to(device)
        #G.weight_init(mean = 0.0, std = 0.02)
        #D.weight_init(mean = 0.0, std = 0.02)
        weights_init(G)
        weights_init(D)
        #G.cuda()
        #D.cuda()

        BCE_loss =nn.BCELoss()

        #print(args.learning_rate)
        G_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        if not os.path.isdir('CelebA_DCGAN_results'):
            os.mkdir('CelebA_DCGAN_results')
        if not os.path.isdir('CelebA_DCGAN_results/Case5'):
            os.mkdir('CelebA_DCGAN_results/Case5')
        if not os.path.isdir('CelebA_DCGAN_results/Case5/Random_results'):
            os.mkdir('CelebA_DCGAN_results/Case5/Random_results')
        if not os.path.isdir('CelebA_DCGAN_results/Case5/Fixed_results'):
            os.mkdir('CelebA_DCGAN_results/Case5/Fixed_results')
        
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []

        print('Training start!')
        start_time = time.time()
        G.train()
        D.train()
        for epoch in range(args.train_epochs):
            D_losses = []
            G_losses = []

            num_iter = 0

            epoch_start_time = time.time()
            for x_, _ in train_loader:

                # train discriminator D
                D.zero_grad()

                mini_batch = x_.size()[0]

                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)

                x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
                D_result = D(x_).squeeze()
                D_real_loss = BCE_loss(D_result, y_real_)

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                z_ = Variable(z_.cuda())
                G_result = G(z_)

                D_result = D(G_result).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_)
                #D_fake_score = D_result.data.mean()

                D_train_loss = D_real_loss + D_fake_loss
                #print(D_train_loss)

                D_train_loss.backward()
                D_optimizer.step()

                D_losses.append(D_train_loss.item())

                # train generator G
                G.zero_grad()

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                z_ = Variable(z_.cuda())

                G_result = G(z_)
                D_result = D(G_result).squeeze()
                G_train_loss = BCE_loss(D_result, y_real_)
                G_train_loss.backward()
                #print(G_train_loss)
                G_optimizer.step()

                G_losses.append(G_train_loss.item())

                num_iter += 1
                #if num_iter % 50 == 0:
                #    print("Training...")
                if num_iter % 10 == 0:
                    print("[num_iter : %d] D_train_loss : %f, G_train_loss : %f" % (num_iter, D_train_loss.item(), G_train_loss.item()))

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time


            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), args.train_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                    torch.mean(torch.FloatTensor(G_losses))))
            p = 'CelebA_DCGAN_results/Case5/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
            fixed_p = 'CelebA_DCGAN_results/Case5/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
            show_result((epoch+1), save=True, path=p, isFix=False)
            show_result((epoch+1), save=True, path=fixed_p, isFix=True)
            train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
            train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

            if (epoch+1) % 5 ==0:
                torch.save(G.state_dict(), "CelebA_DCGAN_results/Case5/generator_param_%d.pkl" % (epoch))
                torch.save(D.state_dict(), "CelebA_DCGAN_results/Case5/discriminator_param_%d.pkl" % (epoch))

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), args.train_epochs, total_ptime))
        print("Training finish!... save training results")
        #torch.save(G.state_dict(), "CelebA_DCGAN_results/Case5/generator_param.pkl")
        #torch.save(D.state_dict(), "CelebA_DCGAN_results/Case5/discriminator_param.pkl")
        with open('CelebA_DCGAN_results/Case5/train_hist.pkl', 'wb') as f:
            pickle.dump(train_hist, f)

        show_train_hist(train_hist, save=True, path='CelebA_DCGAN_results/Case5/CelebA_DCGAN_train_hist.png')

        images = []
        for e in range(args.train_epochs):
            img_name = 'CelebA_DCGAN_results/Case5/Fixed_results/CelebA_DCGAN_' + str(e + 1) + '.png'
            images.append(imageio.imread(img_name))
        imageio.mimsave('CelebA_DCGAN_results/Case5/generation_animation.gif', images, fps=5)