import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_shading

# Base methods for creating convnet


def get_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# SfSNet Models


class ResNetBlock(nn.Module):
    """ Basic building block of ResNet to be used for Normal and Albedo Residual Blocks
    """

    def __init__(self, in_planes, out_planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.res = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, in_planes, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, 3, stride=1, padding=1)
        )

    def forward(self, x):
        residual = x
        out = self.res(x)
        out += residual

        return out


class baseFeaturesExtractions(nn.Module):
    """ Base Feature extraction
    """

    def __init__(self):
        super(baseFeaturesExtractions, self).__init__()
        self.conv1 = get_conv(3, 64, kernel_size=7, padding=3)
        self.conv2 = get_conv(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class NormalResidualBlock(nn.Module):
    """ Net to general Normal from features
    """

    def __init__(self):
        super(NormalResidualBlock, self).__init__()
        self.block1 = ResNetBlock(128, 128)
        self.block2 = ResNetBlock(128, 128)
        self.block3 = ResNetBlock(128, 128)
        self.block4 = ResNetBlock(128, 128)
        self.block5 = ResNetBlock(128, 128)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = F.relu(self.bn1(out))
        return out


class AlbedoResidualBlock(nn.Module):
    """ Net to general Albedo from features
    """

    def __init__(self):
        super(AlbedoResidualBlock, self).__init__()
        self.block1 = ResNetBlock(128, 128)
        self.block2 = ResNetBlock(128, 128)
        self.block3 = ResNetBlock(128, 128)
        self.block4 = ResNetBlock(128, 128)
        self.block5 = ResNetBlock(128, 128)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = F.relu(self.bn1(out))
        return out


class NormalGenerationNet(nn.Module):
    """ Generating Normal
    """

    def __init__(self, device):
        super(NormalGenerationNet, self).__init__()
        # self.upsample = nn.UpsamplingBilinear2d(size=(128, 128), scale_factor=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = get_conv(128, 128, kernel_size=1, stride=1)
        self.conv2 = get_conv(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 2, kernel_size=1)
        self.one = torch.tensor(1).to(device)
        self.epsilon = torch.tensor(1e-4).to(device)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out / torch.maximum(torch.linalg.norm(out, dim=1, keepdim=True), self.one)
        X = out[:,0,:,:]
        Y = out[:,1,:,:]
        Z = torch.sqrt(torch.maximum((1 - X**2 - Y**2), self.epsilon))
        X = torch.unsqueeze(X, 1)
        Y = torch.unsqueeze(Y, 1)
        Z = torch.unsqueeze(Z, 1)
        out = torch.cat((X, Y, Z), dim=1)
        return out


class AlbedoGenerationNet(nn.Module):
    """ Generating Albedo
    """

    def __init__(self):
        super(AlbedoGenerationNet, self).__init__()
        # self.upsample = nn.UpsamplingBilinear2d(size=(128, 128), scale_factor=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = get_conv(128, 128, kernel_size=1, stride=1)
        self.conv2 = get_conv(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class LightEstimator(nn.Module):
    """ Estimate lighting from normal, albedo and conv features
    """

    def __init__(self):
        super(LightEstimator, self).__init__()
        self.conv1 = get_conv(384, 128, kernel_size=1, stride=1)
        self.pool = nn.AvgPool2d(64, stride=1, padding=0)
        self.fc = nn.Linear(128, 28)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        # reshape to batch_size x 128
        out = out.view(-1, 128)
        out = self.fc(out)
        return out


def reconstruct_image(shading, albedo):
    return shading * albedo


class SfsNetPipeline(nn.Module):
    """ SfSNet Pipeline
    """

    def __init__(self, device):
        super(SfsNetPipeline, self).__init__()

        self.conv_model = baseFeaturesExtractions()
        self.normal_residual_model = NormalResidualBlock()
        self.normal_gen_model = NormalGenerationNet(device)
        self.albedo_residual_model = AlbedoResidualBlock()
        self.albedo_gen_model = AlbedoGenerationNet()
        self.light_estimator_model = LightEstimator()
        self.device = device

    def get_face(self, sh, normal, albedo):
        shading = get_shading(normal, sh, self.device)
        recon = reconstruct_image(shading, albedo)
        return recon

    def forward(self, face):
        # Following is training pipeline
        # 1. Pass Image from Conv Model to extract features
        out_features = self.conv_model(face)

        # 2 a. Pass Conv features through Normal Residual
        out_normal_features = self.normal_residual_model(out_features)
        # 2 b. Pass Conv features through Albedo Residual
        out_albedo_features = self.albedo_residual_model(out_features)

        # 3 a. Generate Normal
        predicted_normal = self.normal_gen_model(out_normal_features)
        # 3 b. Generate Albedo
        predicted_albedo = self.albedo_gen_model(out_albedo_features)
        # 3 c. Estimate lighting
        # First, concat conv, normal and albedo features over channels dimension
        all_features = torch.cat(
            (out_features, out_normal_features, out_albedo_features), dim=1)
        # Predict SH
        predicted_sh = self.light_estimator_model(all_features)

        # 4. Generate shading
        out_shading = get_shading(predicted_normal, predicted_sh, self.device)

        # 5. Reconstruction of image
        out_recon = reconstruct_image(out_shading, predicted_albedo)

        return predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon

class SfsNetNormal(nn.Module):
    """ SfSNet Pipeline
    """

    def __init__(self):
        super(SfsNetNormal, self).__init__()

        self.conv_model = baseFeaturesExtractions()
        self.normal_residual_model = NormalResidualBlock()
        self.normal_gen_model = NormalGenerationNet()

    def forward(self, face):
        # Following is training pipeline
        # 1. Pass Image from Conv Model to extract features
        out_features = self.conv_model(face)

        # 2 a. Pass Conv features through Normal Residual
        out_normal_features = self.normal_residual_model(out_features)
        # 3 a. Generate Normal
        predicted_normal = self.normal_gen_model(out_normal_features)
        return predicted_normal


def get_skipnet_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

def get_skipnet_deconv(in_channels, out_channels, kernel_size=3, padding=0, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

class SkipNet_Encoder(nn.Module):
    def __init__(self):
        super(SkipNet_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = get_skipnet_conv(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = get_skipnet_conv(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = get_skipnet_conv(256, 256, kernel_size=4, stride=2, padding=1)
        self.conv5 = get_skipnet_conv(256, 256, kernel_size=4, stride=2, padding=1)
        self.fc256 = nn.Linear(4096, 256)
    
    def get_face(self, sh, normal, albedo):
        shading = get_shading(normal, sh)
        recon   = reconstruct_image(shading, albedo)
        return recon

    def forward(self, x):
        # print('0 ', x.shape )
        out_1 = self.conv1(x)
        # print('1 ', out_1.shape)
        out_2 = self.conv2(out_1)
        # print('2 ', out_2.shape)
        out_3 = self.conv3(out_2)
        # print('3 ', out_3.shape)
        out_4 = self.conv4(out_3)
        # print('4 ', out_4.shape)
        out = self.conv5(out_4)
        # print('5 ', out.shape)
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        out = self.fc256(out)
        return out, out_1, out_2, out_3, out_4
        
class SkipNet_Decoder(nn.Module):
    def __init__(self):
        super(SkipNet_Decoder, self).__init__()
        self.dconv1 = get_skipnet_deconv(256, 256, kernel_size=4, stride=2, padding=1)
        self.dconv2 = get_skipnet_deconv(256, 256, kernel_size=4, stride=2, padding=1)
        self.dconv3 = get_skipnet_deconv(256, 128, kernel_size=4, stride=2, padding=1)
        self.dconv4 = get_skipnet_deconv(128, 64, kernel_size=4, stride=2, padding=1)
        self.dconv5 = get_skipnet_deconv(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv6  = nn.Conv2d(64, 3, kernel_size=1, stride=1)
    
    def forward(self, x, out_1, out_2, out_3, out_4):
        out = self.dconv1(x)
        out += out_4
        out = self.dconv2(out)
        out += out_3
        out = self.dconv3(out)
        out += out_2
        out = self.dconv4(out)
        out += out_1
        out = self.dconv5(out)
        out = self.conv6(out)
        return out

class SkipNet_Normal_Decoder(nn.Module):
    def __init__(self, device):
        super(SkipNet_Normal_Decoder, self).__init__()
        self.dconv1 = get_skipnet_deconv(256, 256, kernel_size=4, stride=2, padding=1)
        self.dconv2 = get_skipnet_deconv(256, 256, kernel_size=4, stride=2, padding=1)
        self.dconv3 = get_skipnet_deconv(256, 128, kernel_size=4, stride=2, padding=1)
        self.dconv4 = get_skipnet_deconv(128, 64, kernel_size=4, stride=2, padding=1)
        self.dconv5 = get_skipnet_deconv(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv6  = nn.Conv2d(64, 2, kernel_size=1, stride=1)
        self.one = torch.tensor(1).to(device)
        self.epsilon = torch.tensor(1e-4).to(device)

    def forward(self, x, out_1, out_2, out_3, out_4):
        out = self.dconv1(x)
        out += out_4
        out = self.dconv2(out)
        out += out_3
        out = self.dconv3(out)
        out += out_2
        out = self.dconv4(out)
        out += out_1
        out = self.dconv5(out)
        out = self.conv6(out)
        out = out / torch.maximum(torch.linalg.norm(out, dim=1, keepdim=True), self.one)
        X = out[:,0,:,:]
        Y = out[:,1,:,:]
        Z = torch.sqrt(torch.maximum((1 - X**2 - Y**2), self.epsilon))
        X = torch.unsqueeze(X, 1)
        Y = torch.unsqueeze(Y, 1)
        Z = torch.unsqueeze(Z, 1)
        out = torch.cat((X, Y, Z), dim=1)
        return out

class SkipNet(nn.Module):
    def __init__(self, device):
        super(SkipNet, self).__init__()
        self.encoder = SkipNet_Encoder()
        self.normal_mlp = nn.Upsample(scale_factor=4, mode='bilinear')
        self.albedo_mlp = nn.Upsample(scale_factor=4, mode='bilinear')
        self.light_decoder = nn.Linear(256, 28)
        self.normal_decoder = SkipNet_Normal_Decoder(device)
        self.albedo_decoder = SkipNet_Decoder()
        self.device = device
 
    def get_face(self, sh, normal, albedo):
        shading = get_shading(normal, sh, self.device)
        recon   = reconstruct_image(shading, albedo)
        return recon
   
    def forward(self, x):
        out, skip_1, skip_2, skip_3, skip_4 = self.encoder(x)
        out_mlp = out.unsqueeze(2)
        out_mlp = out_mlp.unsqueeze(3)
        # print(out_mlp.shape, out.shape)
        out_normal = self.normal_mlp(out_mlp)
        out_albedo = self.albedo_mlp(out_mlp)
        # print(out_normal.shape)
        light = self.light_decoder(out)
        normal = self.normal_decoder(out_normal, skip_1, skip_2, skip_3, skip_4)
        albedo = self.albedo_decoder(out_albedo, skip_1, skip_2, skip_3, skip_4)

        shading = get_shading(normal, light, self.device)
        recon = reconstruct_image(shading, albedo)
        return normal, albedo, light, shading, recon

