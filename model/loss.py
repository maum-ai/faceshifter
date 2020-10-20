import numpy as np

import torch
from torch import nn
from torch.nn import functional as F



class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input).detach()
        else:
            return torch.zeros_like(input).detach()

    def get_zero_tensor(self, input):
        return torch.zeros_like(input).detach()

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)




class AEI_Loss(nn.Module):
    def __init__(self):
        super(AEI_Loss, self).__init__()

        self.att_weight = 10
        self.id_weight = 5
        self.rec_weight = 10

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def att_loss(self, z_att_X, z_att_Y):
        loss = 0
        for i in range(8):
            loss += self.l2(z_att_X[i], z_att_Y[i])
        return 0.5*loss

    def id_loss(self, z_id_X, z_id_Y):
        inner_product = (torch.bmm(z_id_X.unsqueeze(1), z_id_Y.unsqueeze(2)).squeeze())
        return self.l1(torch.ones_like(inner_product), inner_product)

    def rec_loss(self, X, Y, same):
        same = same.unsqueeze(-1).unsqueeze(-1)
        same = same.expand(X.shape)
        X = torch.mul(X, same)
        Y = torch.mul(Y, same)
        return 0.5*self.l2(X, Y)

    def forward(self, X, Y, z_att_X, z_att_Y, z_id_X, z_id_Y, same):

        att_loss = self.att_loss(z_att_X, z_att_Y)
        id_loss = self.id_loss(z_id_X, z_id_Y)
        rec_loss = self.rec_loss(X, Y, same)

        return self.att_weight*att_loss + self.id_weight*id_loss + self.rec_weight*rec_loss, att_loss, id_loss, rec_loss



class HEAR_Loss(nn.Module):
    def __init__(self):
        super(HEAR_Loss, self).__init__()

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def rec_loss(self, X, Y, same):
        same = same.unsqueeze(-1).unsqueeze(-1)
        same = same.expand(X.shape)
        X = torch.mul(X, same)
        Y = torch.mul(Y, same)
        return 0.5 * self.l2(X, Y)

    def id_loss(self, z_id_X, z_id_Y):
        inner_product = (torch.bmm(z_id_X.view(-1, 1, config.z_id_size), z_id_Y.view(-1, config.z_id_size, 1)).squeeze())
        return self.l1(torch.ones_like(inner_product), inner_product)

    def chg_loss(self, Y_hat, Y):
        return self.l1(Y_hat, Y)

    def forward(self, Y_hat, X, Y, z_id_X, z_id_Y, same):
        rec_loss = self.rec_loss(X, Y, same)
        id_loss = self.id_loss(z_id_X, z_id_Y)
        chg_loss = self.chg_loss(Y_hat, Y)
        return rec_loss + id_loss + chg_loss, rec_loss, id_loss, chg_loss