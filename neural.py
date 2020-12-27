from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models

from losses import (
    StyleLoss,
    ContentLoss,
    TVLoss
)


class NeuralStyle():
    def __init__(self):
        # get device -- either cuda or cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.enabled = True
        # layers to get content and style losses from
        self.content_layers_default = ['relu4_2']
        self.style_layers_default = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

    def preprocess(self, image_name, image_size):
        """
        preprocess and open image
        """
        # open image
        image = Image.open(image_name).convert('RGB')
        # reszie
        if type(image_size) is not tuple:
            image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
        Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

        # convert to bgr and normalize
        rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])

        # add a dimension to account for batch size
        tensor = Normalize(rgb2bgr(Loader(image) * 256)).unsqueeze(0)
        return tensor

    def deprocess(self, output_tensor):
        """
        deprocess the image
        """
        Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1])])
        bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0).cpu())) / 256
        output_tensor.clamp_(0, 1)
        Image2PIL = transforms.ToPILImage()
        image = Image2PIL(output_tensor.cpu())
        return image

    def select_model(self, model_selection='vgg19'):
        # models = vgg16, vgg19
        if model_selection == 'vgg19':
            self.cnn = models.vgg19(pretrained=True).features
            self.model_dict = {
            'conv': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'],
            'relu': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4', 'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4', 'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4'],
            'pool': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
            }
        else:
            self.cnn = models.vgg16(pretrained=True).features
            self.model_dict = {
            'conv': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'],
            'relu': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3'],
            'pool': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
            }

    def set_style_scale(self, scale):
        self.style_scale = scale

    def plot_content_then_style(self):
        """
        plot the content, then the style image
        """
        # plot content image
        plt.figure()
        plt.imshow(self.deprocess(self.content_img))
        plt.show()

        # plot style image
        plt.figure()
        plt.imshow(self.deprocess(self.style_img))
        plt.show()

    def get_img(self, style_dir='style', content_dir='content',
                style_img_name=None, content_img_name=None,
                img_dim=314):
        size = img_dim
        list_img = os.listdir(content_dir)

        if content_img_name is None:
            full_img = np.random.choice(list_img)
        else:
            if content_img_name not in list_img:
                raise FileNotFoundError('Img File does not exist!')
            full_img = content_img_name
        full_img_string = os.path.join(content_dir, full_img)
        # style_size = size * self.style_scale

        self.content_img = self.preprocess(full_img_string, size).to(self.device)

        list_style_img = os.listdir(style_dir)

        if style_img_name is None:
            style_string = np.random.choice(list_style_img)
        else:
            if style_img_name not in list_style_img:
                raise FileNotFoundError('Style image does not exist!')
            style_string = style_img_name

        # get style image
        full_style_string = os.path.join(style_dir, style_string)

        self.style_img = self.preprocess(full_style_string, size).to(self.device)

    def create_white_noise(self):
        self.input_img = self.content_img.clone().to(self.device)

        # white noise!!
        B, C, H, W = self.content_img.size()
        self.input_img = torch.randn(B, C, H, W).mul(0.001).to(self.device)

    def get_input_optimizer(self, input_img, name='lbfgs', iterations=1000):
        # this line to show that input is a parameter that requires a gradient
        if name == 'lbfgs':
            optim_state = {
                    'max_iter': iterations,
                    'tolerance_change': -1,
                    'tolerance_grad': -1,
                }
            self.loop_val = 1
            self.optimizer = optim.LBFGS([input_img], **optim_state)
            print('Running optimization with LBFGS (as in the paper)')

        else:
            print("Running optimization with ADAM")
            self.optimizer = optim.Adam([input_img], lr=0.01)
            self.loop_val = 1
        return self.optimizer, self.loop_val

    def run_style_transfer(self, style_weight=500, content_weight=5,
                           iterations=1000):
        """
        Run the style transfer.
        """
        print('Building the style transfer model..')
        self.create_white_noise()
        model, style_losses, content_losses, tv_losses = self.get_style_model_and_losses(self.model_dict,
                            self.style_img, self.content_img)
        self.input_img = nn.Parameter(self.input_img)
        for param in model.parameters():
            param.requires_grad = False
        print('Optimizing..')
        run = [0]

        def closure():
            # correct the values of updated input image to be in (0,1)
            # input_img.data.clamp_(0, 1)

            run[0] += 1
            optimizer.zero_grad()
            model(self.input_img)
            style_score = 0
            content_score = 0
            tv_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            for tl in tv_losses:
                tv_score += tl.loss
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score + tv_score

            loss.backward()
            print('iteration no.', run[0])
            if run[0] % 100 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                os.makedirs('generated', exist_ok=True)
                i = run[0]
                self.deprocess(self.input_img).save(f'generated/out_{i}.png')

            return loss

        optimizer, loop_val = self.get_input_optimizer(self.input_img, iterations=iterations)
        print(optimizer)
        while run[0] <= loop_val:
            optimizer.step(closure)
            # a last correction...
            # input_img.data.clamp_(0, 1)

        return self.input_img

    def get_style_model_and_losses(self, model_dict,
                                   style_img, content_img):
        """
        get the losses.
        """
        self.cnn = copy.deepcopy(self.cnn)
        self.cnn = self.cnn.to(self.device)
        c_idx = 0
        r_idx = 0
        p_idx = 0
        # do some normalization!

        # list of losses in layers:
        content_losses = []
        style_losses = []
        tv_losses = []

        model = nn.Sequential()
        i = 0
        tv_mod = TVLoss(1e-3)
        model.add_module(str(len(model)), tv_mod)
        tv_losses.append(tv_mod)

        for layer in self.cnn.children():

            if isinstance(layer, nn.Conv2d):
                i += 1
                name = model_dict['conv'][c_idx]
                c_idx += 1
            elif isinstance(layer, nn.ReLU):
                name = model_dict['relu'][r_idx]
                r_idx += 1
                layer = nn.ReLU(inplace=True)

            elif isinstance(layer, nn.MaxPool2d):
                name = model_dict['pool'][p_idx]
                # layer = nn.AvgPool2d(kernel_size=2, stride=2)
                p_idx += 1

            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'

            else:
                layer_name = layer.__class__.__name__
                raise RuntimeError(f'Unrecognized Layer: {layer_name}')

            model.add_module(name, layer)

            if name in self.content_layers_default:
                content_loss = ContentLoss()
                model.add_module(f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers_default:
                # get feature maps from style
                style_loss = StyleLoss()
                model.add_module(f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        # if there is extra non-needed layers.
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]
        # rip through the model, getting the REAL feature maps for style and content.
        for module in style_losses:
            module.mode = "capture"
        model(style_img)
        for module in style_losses:
            module.mode = "none"
        for module in content_losses:
            module.mode = "capture"
        model(content_img)
        for module in style_losses:
            module.mode = "loss"
        for module in content_losses:
            module.mode = "loss"
        return model, style_losses, content_losses, tv_losses


if __name__ == '__main__':
    # generate images!
    neural_style_system = NeuralStyle()
    # get randon style+content image
    neural_style_system.get_img(content_img_name='ma3.jpg',
                                style_img_name='flowercarrier.jpg')
    neural_style_system.plot_content_then_style()

    # select the model (by default, we select the vgg-19 model)
    neural_style_system.select_model()

    neural_style_system.run_style_transfer(style_weight=1000, content_weight=5)
