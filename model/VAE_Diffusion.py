import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out

class NewConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NewConvModule, self).__init__()

        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.cbam = CBAM(out_channels)

        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.adjust_channels(x)  # 调整通道数

        out = self.conv1x1_1(x)
        out = F.relu(out)

        out = self.conv3x1(out)
        out = F.relu(out)

        out = self.conv1x3(out)
        out = F.relu(out)

        out = self.conv1x1_2(out)
        out = F.relu(out)

        out = self.cbam(out)

        out += identity
        return out

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class Encoder(nn.Module):
    def __init__(self, t_length=2, n_channel=3, latent_dim=128):
        super(Encoder, self).__init__()

        def Basic(intInput, intOutput):
            return nn.Sequential(
                nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(intOutput),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(intOutput),
                nn.ReLU(inplace=False)
            )

        self.new_conv_module = NewConvModule(n_channel * (t_length - 1), 64)

        self.moduleConv1 = Basic(64, 64)
        self.modulePool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(256, 512)
        self.modulePool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convLSTM1 = ConvLSTMCell(input_dim=512, hidden_dim=512, kernel_size=3, bias=True)
        self.convLSTM2 = ConvLSTMCell(input_dim=512, hidden_dim=512, kernel_size=3, bias=True)

        self.fc_mu = nn.Linear(512 * (16 * 16), latent_dim)
        self.fc_logvar = nn.Linear(512 * (16 * 16), latent_dim)

    def forward(self, x):
        x = self.new_conv_module(x)

        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        batch_size, _, height, width = tensorPool4.size()
        h, c = self.convLSTM1.init_hidden(batch_size, (height, width))
        tensorLSTM1, _ = self.convLSTM1(tensorPool4, (h, c))
        tensorLSTM2, _ = self.convLSTM2(tensorLSTM1, (h, c))

        tensor_flat = tensorLSTM2.view(tensorLSTM2.size(0), -1)

        mu = self.fc_mu(tensor_flat)
        logvar = self.fc_logvar(tensor_flat)

        return mu, logvar, tensorLSTM2, tensorPool3, tensorPool2, tensorPool1

class Decoder(nn.Module):
    def __init__(self, t_length=2, n_channel=3, latent_dim=128):
        super(Decoder, self).__init__()

        def Basic(intInput, intOutput):
            return nn.Sequential(
                nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(intOutput),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(intOutput),
                nn.ReLU(inplace=False)
            )

        def Upsample(nc, intOutput):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=2, stride=2),
                nn.BatchNorm2d(intOutput),
                nn.ReLU(inplace=False)
            )

        self.fc_dec = nn.Linear(latent_dim, 512 * (16 * 16))

        self.convLSTM3 = ConvLSTMCell(input_dim=512, hidden_dim=512, kernel_size=3, bias=True)
        self.convLSTM4 = ConvLSTMCell(input_dim=512, hidden_dim=512, kernel_size=3, bias=True)

        self.moduleUpsample4 = Upsample(512, 256)
        self.moduleDeconv3 = Basic(512, 256)

        self.moduleUpsample3 = Upsample(256, 128)
        self.moduleDeconv2 = Basic(256, 128)

        self.moduleUpsample2 = Upsample(128, 64)
        self.moduleDeconv1 = Basic(128, 64)

        self.moduleUpsample1 = Upsample(64, 32)
        self.moduleFinalConv = nn.Conv2d(32, n_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        mu, logvar, tensorPool4, tensorPool3, tensorPool2, tensorPool1 = x

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        tensor_dec = self.fc_dec(z)
        tensor_dec = tensor_dec.view(tensor_dec.size(0), 512, 16, 16)

        batch_size, _, height, width = tensor_dec.size()
        h, c = self.convLSTM3.init_hidden(batch_size, (height, width))
        tensorLSTM3, _ = self.convLSTM3(tensor_dec, (h, c))
        tensorLSTM4, _ = self.convLSTM4(tensorLSTM3, (h, c))

        tensorUpsample4 = self.moduleUpsample4(tensorLSTM4)
        tensorCat4 = torch.cat((tensorUpsample4, tensorPool3), 1)
        tensorDeconv3 = self.moduleDeconv3(tensorCat4)

        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        tensorCat3 = torch.cat((tensorUpsample3, tensorPool2), 1)
        tensorDeconv2 = self.moduleDeconv2(tensorCat3)

        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        tensorCat2 = torch.cat((tensorUpsample2, tensorPool1), 1)
        tensorDeconv1 = self.moduleDeconv1(tensorCat2)

        tensorUpsample1 = self.moduleUpsample1(tensorDeconv1)
        output = self.moduleFinalConv(tensorUpsample1)

        return output

class convAE(nn.Module):
    def __init__(self, n_channel=3, t_length=2, latent_dim=128):
        super(convAE, self).__init__()

        self.encoder = Encoder(t_length, n_channel, latent_dim)
        self.decoder = Decoder(t_length, n_channel, latent_dim)

    def forward(self, x):
        mu, logvar, *encoder_features = self.encoder(x)
        output = self.decoder((mu, logvar, *encoder_features))
        return output, mu, logvar

class DiffusionModel(nn.Module):
    def __init__(self, steps=1000, beta_start=0.0001, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.steps = steps
        # 线性变化
        #self.betas = torch.linspace(beta_start, beta_end, steps)

        # 指数变化
        #self.betas = beta_start * (beta_end / beta_start) ** (torch.arange(steps) / (steps - 1))

        # 余弦变化
        self.betas = beta_start + (beta_end - beta_start) * (1 - torch.cos(torch.linspace(0, math.pi, steps))) / 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t]
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

    def reverse_process(self, x):
        for t in reversed(range(self.steps)):
            noise = torch.randn_like(x)
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            x = (x - (1 - alpha_t) * noise) / torch.sqrt(alpha_t)
            x = x + torch.sqrt(beta_t) * noise
        return x

class convAEDiffusion(nn.Module):
    def __init__(self, n_channel=3, t_length=2, latent_dim=128):
        super(convAEDiffusion, self).__init__()
        self.encoder = Encoder(t_length, n_channel, latent_dim)
        self.decoder = Decoder(t_length, n_channel, latent_dim)
        self.diffusion_model = DiffusionModel()

    def forward(self, x):
        mu, logvar, *encoder_features = self.encoder(x)
        initial_output = self.decoder((mu, logvar, *encoder_features))

        # Apply diffusion model for refinement
        refined_output = self.diffusion_model.reverse_process(initial_output)

        return refined_output, mu, logvar