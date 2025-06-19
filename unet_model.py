import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(x1, size=g1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetAttention, self).__init__()

        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bridge
        self.bottleneck = DoubleConv(512, 1024)

        # Attention Gates
        self.attn_gate3 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.attn_gate2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.attn_gate1 = AttentionGate(F_g=128, F_l=128, F_int=64)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder + Attention
        up4 = self.upconv4(bottleneck)
        att3 = self.attn_gate3(g=up4, x=enc4)
        up4 = torch.cat([up4, att3], dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.upconv3(dec4)
        att2 = self.attn_gate2(g=up3, x=enc3)
        up3 = torch.cat([up3, att2], dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.upconv2(dec3)
        att1 = self.attn_gate1(g=up2, x=enc2)
        up2 = torch.cat([up2, att1], dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.upconv1(dec2)
        up1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.decoder1(up1)

        return self.final_conv(dec1)
    


# --- 3. Definiciones de los Módulos (R_ELAN_Block Corregido) ---
class R_ELAN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=2, scaling_factor=1.0): # <-- CORRECCIÓN
        super(R_ELAN_Block, self).__init__()
        self.scaling_factor = scaling_factor
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        mid_channels = out_channels // expansion_factor
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True)
        )
        
        self.aggregate_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual = x
        x_init = self.initial_conv(x)
        out1 = self.branch1(x_init)
        out2 = self.branch2(x_init)
        x_agg = self.aggregate_conv(torch.cat([out1, out2], dim=1))

        if residual.size(1) != x_agg.size(1):
             residual = self.initial_conv(residual)
        
        return x_agg + self.scaling_factor * residual

class ASPP_Block(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP_Block, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.aspp_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
            ) for rate in rates
        ])
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(rates)), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )

    def forward(self, x):
        out1 = self.conv1x1(x)
        aspp_outs = [conv(x) for conv in self.aspp_convs]
        img_pool_out = F.interpolate(self.image_pooling(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        concatenated_features = torch.cat([out1] + aspp_outs + [img_pool_out], dim=1)
        output = self.final_conv(concatenated_features)
        return output

class UNet_RELAN_ASPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_RELAN_ASPP, self).__init__()
        def r_elan_block(in_ch, out_ch): return R_ELAN_Block(in_ch, out_ch)

        self.enc1, self.pool1 = r_elan_block(in_channels, 64), nn.MaxPool2d(2)
        self.enc2, self.pool2 = r_elan_block(64, 128), nn.MaxPool2d(2)
        self.enc3, self.pool3 = r_elan_block(128, 256), nn.MaxPool2d(2)
        self.enc4, self.pool4 = r_elan_block(256, 512), nn.MaxPool2d(2)

        self.bottleneck_conv = r_elan_block(512, 1024)
        self.aspp = ASPP_Block(1024, 1024, rates=[6, 12, 18])

        self.upconv4, self.dec4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2), r_elan_block(1024, 512)
        self.upconv3, self.dec3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), r_elan_block(512, 256)
        self.upconv2, self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), r_elan_block(256, 128)
        self.upconv1, self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), r_elan_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.aspp(self.bottleneck_conv(self.pool4(e4)))

        d4 = self.upconv4(b)
        if d4.shape[-2:] != e4.shape[-2:]: d4 = F.interpolate(d4, size=e4.shape[-2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.upconv3(d4)
        if d3.shape[-2:] != e3.shape[-2:]: d3 = F.interpolate(d3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        if d2.shape[-2:] != e2.shape[-2:]: d2 = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        if d1.shape[-2:] != e1.shape[-2:]: d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final_conv(d1)


## Modelo:class AttentionGate(nn.Module):
### **Modelo: UNet con Atención (`UNetAttention`)**

# class AttentionGate(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(AttentionGate, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)

#         # Interpolar x1 si las dimensiones espaciales no coinciden (ajuste para la concatenación)
#         if g1.shape[2:] != x1.shape[2:]:
#             x1 = F.interpolate(x1, size=g1.shape[2:], mode='bilinear', align_corners=True)

#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class UNetAttention(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3): # out_channels=3 para 3 clases (fondo, arrugas, manchas)
#         super(UNetAttention, self).__init__()

#         # Encoder
#         self.encoder1 = DoubleConv(in_channels, 64)
#         self.pool1 = nn.MaxPool2d(2)

#         self.encoder2 = DoubleConv(64, 128)
#         self.pool2 = nn.MaxPool2d(2)

#         self.encoder3 = DoubleConv(128, 256)
#         self.pool3 = nn.MaxPool2d(2)

#         self.encoder4 = DoubleConv(256, 512)
#         self.pool4 = nn.MaxPool2d(2)

#         # Bridge
#         self.bottleneck = DoubleConv(512, 1024)

#         # Attention Gates (g: from decoder path, x: from encoder skip connection)
#         self.attn_gate3 = AttentionGate(F_g=512, F_l=512, F_int=256)
#         self.attn_gate2 = AttentionGate(F_g=256, F_l=256, F_int=128)
#         self.attn_gate1 = AttentionGate(F_g=128, F_l=128, F_int=64)

#         # Decoder
#         self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.decoder4 = DoubleConv(1024, 512)

#         self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.decoder3 = DoubleConv(512, 256)

#         self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.decoder2 = DoubleConv(256, 128)

#         self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.decoder1 = DoubleConv(128, 64)

#         self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))

#         # Bottleneck
#         bottleneck = self.bottleneck(self.pool4(enc4))

#         # Decoder + Attention
#         up4 = self.upconv4(bottleneck)
#         if up4.shape[2:] != enc4.shape[2:]: # Ajustar tamaño si ConvTranspose no coincide exactamente
#              up4 = F.interpolate(up4, size=enc4.shape[2:], mode='bilinear', align_corners=True)
#         att3 = self.attn_gate3(g=up4, x=enc4)
#         up4 = torch.cat([up4, att3], dim=1)
#         dec4 = self.decoder4(up4)

#         up3 = self.upconv3(dec4)
#         if up3.shape[2:] != enc3.shape[2:]:
#              up3 = F.interpolate(up3, size=enc3.shape[2:], mode='bilinear', align_corners=True)
#         att2 = self.attn_gate2(g=up3, x=enc3)
#         up3 = torch.cat([up3, att2], dim=1)
#         dec3 = self.decoder3(up3)

#         up2 = self.upconv2(dec3)
#         if up2.shape[2:] != enc2.shape[2:]:
#              up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
#         att1 = self.attn_gate1(g=up2, x=enc2)
#         up2 = torch.cat([up2, att1], dim=1)
#         dec2 = self.decoder2(up2)

#         up1 = self.upconv1(dec2)
#         if up1.shape[2:] != enc1.shape[2:]:
#              up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=True)
#         up1 = torch.cat([up1, enc1], dim=1)
#         dec1 = self.decoder1(up1)

#         return self.final_conv(dec1)
# --- Fin del modelo ---
# Cargar el modelo