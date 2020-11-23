import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import resnet101
import pytorch_lightning as pl

from model.AEINet import ADDGenerator, MultilevelAttributesEncoder
from model.MultiScaleDiscriminator import MultiscaleDiscriminator

from model.loss import GANLoss, AEI_Loss

from dataset import *

class AEINet(pl.LightningModule):
    def __init__(self, hp):
        super(AEINet, self).__init__()
        self.hp = hp

        self.G = ADDGenerator(hp.arcface.vector_size)
        self.E = MultilevelAttributesEncoder()
        self.D = MultiscaleDiscriminator(3)

        self.Z = resnet101(num_classes=256)
        self.Z.load_state_dict(torch.load(hp.arcface.chkpt_path, map_location='cpu'))

        self.Loss_GAN = GANLoss()
        self.Loss_E_G = AEI_Loss()


    def forward(self, target_img, source_img):
        z_id = self.Z(F.interpolate(source_img, size=112, mode='bilinear'))
        z_id = F.normalize(z_id)
        z_id = z_id.detach()

        feature_map = self.E(target_img)

        output = self.G(z_id, feature_map)

        output_z_id = self.Z(F.interpolate(output, size=112, mode='bilinear'))
        output_z_id = F.normalize(output_z_id)
        output_feature_map = self.E(output)
        return output, z_id, output_z_id, feature_map, output_feature_map


    def training_step(self, batch, batch_idx, optimizer_idx):
        target_img, source_img, same = batch

        if optimizer_idx == 0:
            output, z_id, output_z_id, feature_map, output_feature_map = self(target_img, source_img)

            self.generated_img = output

            output_multi_scale_val = self.D(output)
            loss_GAN = self.Loss_GAN(output_multi_scale_val, True, for_discriminator=False)
            loss_E_G, loss_att, loss_id, loss_rec = self.Loss_E_G(target_img, output, feature_map, output_feature_map, z_id,
                                                             output_z_id, same)

            loss_G = loss_E_G + loss_GAN

            self.logger.experiment.add_scalar("Loss G", loss_G.item(), self.global_step)
            self.logger.experiment.add_scalar("Attribute Loss", loss_att.item(), self.global_step)
            self.logger.experiment.add_scalar("ID Loss", loss_id.item(), self.global_step)
            self.logger.experiment.add_scalar("Reconstruction Loss", loss_rec.item(), self.global_step)
            self.logger.experiment.add_scalar("GAN Loss", loss_GAN.item(), self.global_step)

            return loss_G

        else:
            multi_scale_val = self.D(target_img)
            output_multi_scale_val = self.D(self.generated_img.detach())

            loss_D_fake = self.Loss_GAN(multi_scale_val, True)
            loss_D_real = self.Loss_GAN(output_multi_scale_val, False)

            loss_D = loss_D_fake + loss_D_real

            self.logger.experiment.add_scalar("Loss D", loss_D.item(), self.global_step)
            return loss_D

    def validation_step(self, batch, batch_idx):
        target_img, source_img, same = batch

        output, z_id, output_z_id, feature_map, output_feature_map = self(target_img, source_img)

        self.generated_img = output

        output_multi_scale_val = self.D(output)
        loss_GAN = self.Loss_GAN(output_multi_scale_val, True, for_discriminator=False)
        loss_E_G, loss_att, loss_id, loss_rec = self.Loss_E_G(target_img, output, feature_map, output_feature_map,
                                                              z_id, output_z_id, same)
        loss_G = loss_E_G + loss_GAN
        return {"loss": loss_G, 'target': target_img[0].cpu(), 'source': source_img[0].cpu(),  "output": output[0].cpu(), }

    def validation_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        validation_image = []
        for x in outputs:
            validation_image = validation_image + [x['target'], x['source'], x["output"]]
        validation_image = torchvision.utils.make_grid(validation_image, nrow=3)

        self.logger.experiment.add_scalar("Validation Loss", loss.item(), self.global_step)
        self.logger.experiment.add_image("Validation Image", validation_image, self.global_step)

        return {"loss": loss, "image": validation_image, }


    def configure_optimizers(self):
        lr_g = self.hp.model.learning_rate_E_G
        lr_d = self.hp.model.learning_rate_D
        b1 = self.hp.model.beta1
        b2 = self.hp.model.beta2

        opt_g = torch.optim.Adam(list(self.G.parameters()) + list(self.E.parameters()), lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            ])
        dataset = AEI_Dataset(self.hp.data.dataset_dir, transform=transform)
        return DataLoader(dataset, batch_size=self.hp.model.batch_size, num_workers=self.hp.model.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
        ])
        dataset = AEI_Val_Dataset(self.hp.data.valset_dir, transform=transform)
        return DataLoader(dataset, batch_size=1, shuffle=False)
