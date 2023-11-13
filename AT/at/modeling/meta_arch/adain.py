import os
import torch

import torch.nn as nn
from torchvision import models

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat, fc1,fc2):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    mixed_style_mean = torch.cat((style_mean,content_mean),1).squeeze(2).squeeze(2)
    mixed_style_std = torch.cat((style_std,content_std),1).squeeze(2).squeeze(2)

    new_style_mean = (fc1(mixed_style_mean)).unsqueeze(2).unsqueeze(2)
    new_style_std = (fc2(mixed_style_std)).unsqueeze(2).unsqueeze(2)
    return normalized_feat * new_style_std.expand(size) + new_style_mean.expand(size)



decoder = nn.Sequential(
    # vgg[:19]
    nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
)


vgg = models.vgg16()
vgg=nn.Sequential(*list(vgg.features._modules.values())[:-1])
vgg[4].ceil_mode=True
vgg[9].ceil_mode=True
vgg[16].ceil_mode=True

fc1 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(inplace=True),nn.Linear(512,512))
fc2 = nn.Sequential(nn.Linear(1024,512),nn.ReLU(inplace=True),nn.Linear(512,512))


class AdainNet(nn.Module):
    def __init__(self, encoder, decoder,fc1,fc2):
        super(AdainNet, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:2])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[2:7])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[7:12])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[12:])  # relu3_1 -> relu4_1
        dec_layers = list(decoder.children())
        self.dec_1 = nn.Sequential(*dec_layers[:7])  # input -> relu1_1
        self.dec_2 = nn.Sequential(*dec_layers[7:12])  # relu1_1 -> relu2_1
        self.dec_3 = nn.Sequential(*dec_layers[12:17])  # relu2_1 -> relu3_1
        self.dec_4 = nn.Sequential(*dec_layers[17:])  # relu3_1 -> relu4_1
        self.mse_loss = nn.MSELoss()
        self.fc1 = fc1
        self.fc2 = fc2
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def decode_with_intermediate(self,input):
        results=[input]
        for i in range(4):
            func = getattr(self, 'dec_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        # assert (input.size() == target.size())

        if not input.size() == target.size():
            target = target[:,:,:input.shape[2],:input.shape[3]]
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def calc_content_constraint_loss(self,input,target):
        # assert (input.size() == target.size())

        if not input.size() == target.size():
            target = target[:,:,:input.shape[2],:input.shape[3]]
        # print(input.shape)
        # print(target.shape)
        return self.mse_loss(input, target)
    
    def forward(self, content, style, flag=0, alpha=1.0):
        assert 0 <= alpha <= 1
        if flag==0:
            style_feats = self.encode_with_intermediate(style)
            content_feats = self.encode_with_intermediate(content)
            # print(f"style_feats:{style_feats[-1].shape}")
            # print(f"content_feats:{content_feats[-1].shape}")
            
            t = adaptive_instance_normalization(content_feats[-1], style_feats[-1],self.fc1,self.fc2)
            t = alpha * t + (1 - alpha) * content_feats[-1]

            g_ts = self.decode_with_intermediate(t)
            g_t_feats = self.encode_with_intermediate(g_ts[-1])
            loss_c = self.calc_content_loss(g_t_feats[-1], t.detach())
            

            deco_ts = self.decode_with_intermediate(content_feats[-1])
            
            # print(f"content_feats:{content_feats[0].shape}")
            # print(f"deco_ts[-1]:{deco_ts[-1].shape}")
            # print(f"deco_ts[-2]:{deco_ts[-2].shape}")

            loss_const = self.calc_content_constraint_loss(content_feats[0], deco_ts[-2])
            for i in range(1, 3):
                loss_const += self.calc_content_constraint_loss(content_feats[i], deco_ts[-(i+2)])
            loss_const += self.calc_content_constraint_loss(content,deco_ts[-1])
            return loss_c,loss_const
        
        elif flag==1:
            style_feats = self.encode_with_intermediate(style)
            content_feats=self.encode_with_intermediate(content)
            t = adaptive_instance_normalization(content_feats[-1], style_feats[-1],self.fc1,self.fc2)
            t = alpha * t + (1 - alpha) * content_feats[-1]
            g_ts = self.decode_with_intermediate(t)
            g_t_feats = self.encode_with_intermediate(g_ts[-1])

            
            loss_s_1 = self.calc_style_loss(g_t_feats[0], style_feats[0])
            for i in range(1, 4):
                loss_s_1 += self.calc_style_loss(g_t_feats[i], style_feats[i])
            loss_s_2 = self.calc_style_loss(g_t_feats[0], content_feats[0])
            for i in range(1, 4):
                loss_s_2 += self.calc_style_loss(g_t_feats[i], content_feats[i])

            return loss_s_1,loss_s_2

        elif flag == 2:
            style_feats = self.encode_with_intermediate(style)
            content_feats=self.encode_with_intermediate(content)
            t = adaptive_instance_normalization(content_feats[-1], style_feats[-1],self.fc1,self.fc2)
            t = alpha * t + (1 - alpha) * content_feats[-1]
            g_ts = self.decode_with_intermediate(t)
            return g_ts[-1]
