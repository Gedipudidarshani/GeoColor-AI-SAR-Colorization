import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundedHomomorphicDespeckle(nn.Module):
    """Stage 1: log-domain (homomorphic) residual despeckler.

    Multiplicative model Y = X * F becomes additive in the log domain,
    log Y = log X + log F. The CNN predicts the speckle component log F
    (bounded by tanh) and subtracts it: log X_hat = log Y - log F_hat.
    """
    def __init__(self, in_channels=1, num_features=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features)
            ) for _ in range(2)
        ])
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x_pos = torch.clamp(x, 0.0, 1.0) + 0.01
        log_y = torch.log(x_pos)

        feat = F.leaky_relu(self.conv_in(log_y), 0.2)
        for block in self.blocks:
            feat = F.leaky_relu(feat + block(feat), 0.2)

        pred_noise = self.conv_out(feat) * 0.5
        clean_log = log_y - pred_noise
        clean_sar = torch.exp(clean_log) - 0.01
        return torch.clamp(clean_sar, 0.0, 1.0)


class IdentityDespeckle(nn.Module):
    """Used by baselines / ablations without a learned despeckling stage."""
    def forward(self, x):
        return x


class DecoderBlock(nn.Module):
    """Bilinear upsampling + conv + (MC) spatial dropout."""
    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dropout_p = dropout_p
        # When True, dropout stays active in eval mode (Monte Carlo dropout).
        self.mc_dropout = False

    def forward(self, x):
        x = self.up(x)
        return F.dropout2d(x, p=self.dropout_p, training=self.training or self.mc_dropout)


class ResNetEncoderStage(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class SARColorizerGenerator(nn.Module):
    """Two-stage generator: homomorphic despeckler -> 4-level U-Net colorizer.

    use_despeckle=False gives a plain U-Net (Pix2Pix-style generator) with the
    same colorizer, used for baselines and ablations.
    encoder='resnet18' replaces the from-scratch encoder by an ImageNet-pretrained
    ResNet-18 (stem, layer1-3). n_classes>0 adds FiLM conditioning on the terrain
    class; index n_classes is the learned "unknown terrain" embedding.
    """
    def __init__(self, in_channels=1, out_channels=3, use_despeckle=True, encoder="unet", n_classes=0,
                 pretrained=True):
        super().__init__()
        self.despeckle = BoundedHomomorphicDespeckle(in_channels=in_channels) if use_despeckle else IdentityDespeckle()
        self.encoder = encoder
        self.n_classes = n_classes

        if encoder == "unet":
            # Encoder: 256 -> 128 -> 64 -> 32 -> 16
            c1, c2, c3, c4 = 64, 128, 256, 512
            self.e1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
            self.e2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
            self.e3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
            self.e4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        elif encoder == "resnet18":
            import torchvision
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            r = torchvision.models.resnet18(weights=w)
            c1, c2, c3, c4 = 64, 64, 128, 256
            self.e1 = ResNetEncoderStage(r.conv1, r.bn1, r.relu)        # H/2
            self.e2 = ResNetEncoderStage(r.maxpool, r.layer1)           # H/4
            self.e3 = ResNetEncoderStage(r.layer2)                      # H/8
            self.e4 = ResNetEncoderStage(r.layer3)                      # H/16
            self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            raise ValueError(encoder)

        self.mid = nn.Sequential(
            nn.Conv2d(c4, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512)
        )

        self.d4 = DecoderBlock(512, 256, dropout_p=0.15)
        self.d3 = DecoderBlock(256 + c3, 128, dropout_p=0.10)
        self.d2 = DecoderBlock(128 + c2, 64, dropout_p=0.05)
        self.d1 = DecoderBlock(64 + c1, 32, dropout_p=0.0)

        self.final = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

        if n_classes > 0:
            # FiLM (Perez et al., 2018): per-class scale/shift on bottleneck + decoder features.
            # Zero-initialised, so training starts from the unconditioned network.
            self.film_dims = [512, 256, 128, 64]
            self.film = nn.Embedding(n_classes + 1, 2 * sum(self.film_dims))
            nn.init.zeros_(self.film.weight)

    def set_mc_dropout(self, enabled: bool):
        for m in self.modules():
            if isinstance(m, DecoderBlock):
                m.mc_dropout = enabled

    def _film(self, cls, batch, device):
        if self.n_classes == 0:
            return None
        if cls is None:
            cls = torch.full((batch,), self.n_classes, dtype=torch.long, device=device)
        p = self.film(cls)
        out, i = [], 0
        for d in self.film_dims:
            g, b = p[:, i:i + d], p[:, i + d:i + 2 * d]
            out.append((g[:, :, None, None], b[:, :, None, None]))
            i += 2 * d
        return out

    @staticmethod
    def _film_mod(x, gb):
        return x if gb is None else x * (1 + gb[0]) + gb[1]

    def forward(self, x, cls=None):
        clean_sar = self.despeckle(x)
        film = self._film(cls, x.shape[0], x.device)
        f = film if film is not None else [None] * 4

        enc_in = clean_sar
        if self.encoder == "resnet18":
            enc_in = (clean_sar.repeat(1, 3, 1, 1) - self.img_mean) / self.img_std
        e1 = self.e1(enc_in)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self._film_mod(self.mid(e4), f[0])

        d4 = self._film_mod(self.d4(b), f[1])
        d3 = self._film_mod(self.d3(torch.cat([d4, e3], dim=1)), f[2])
        d2 = self._film_mod(self.d2(torch.cat([d3, e2], dim=1)), f[3])
        d1 = self.d1(torch.cat([d2, e1], dim=1))

        out_rgb = self.final(torch.cat([d1, clean_sar], dim=1))
        return clean_sar, out_rgb

    @torch.no_grad()
    def predict_with_epistemic_uncertainty(self, x, passes=10, cls=None):
        """Monte Carlo dropout inference (Gal & Ghahramani, 2016).

        BatchNorm stays in eval mode (running statistics); only the decoder
        dropout layers are stochastic. Returns (clean_sar, mean_rgb, variance)
        where variance is the per-pixel predictive variance averaged over RGB.
        With passes == 1 the prediction is deterministic and the uncertainty
        map is all zeros (no uncertainty can be estimated from one sample).
        """
        self.eval()
        if passes <= 1:
            self.set_mc_dropout(False)
            clean_sar, rgb = self.forward(x, cls)
            return clean_sar, rgb, torch.zeros_like(clean_sar)

        self.set_mc_dropout(True)
        preds = []
        clean_sar = None
        for _ in range(passes):
            clean_sar, rgb = self.forward(x, cls)
            preds.append(rgb)
        self.set_mc_dropout(False)

        stacked = torch.stack(preds, dim=0)
        mean_rgb = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=True).mean(dim=1, keepdim=True)
        return clean_sar, mean_rgb, variance


class PatchGANDiscriminator(nn.Module):
    """70x70-style PatchGAN on concatenated SAR + optical [B, 4, H, W]."""
    def __init__(self, in_channels=4):
        super().__init__()
        def block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, sar, optical):
        return self.net(torch.cat([sar, optical], dim=1))


def lee_filter(x, window=7, looks=40.0):
    """Classical Lee (1980) adaptive speckle filter, batched torch version.

    x: [B,1,H,W] in [0,1]. Speckle variance coefficient Cu^2 = 1/L. L=40 is the
    median ENL measured on homogeneous blocks of the training scenes.
    """
    pad = window // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    mean = F.avg_pool2d(xp, window, stride=1)
    sq_mean = F.avg_pool2d(xp * xp, window, stride=1)
    var_y = (sq_mean - mean * mean).clamp_min(0.0)
    cu2 = 1.0 / looks
    var_x = ((var_y - cu2 * mean * mean) / (1.0 + cu2)).clamp_min(0.0)
    k = var_x / (var_y + 1e-8)
    return (mean + k * (x - mean)).clamp(0.0, 1.0)
