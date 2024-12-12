import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb

from ._base import Distiller


def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


class ReviewKD_RGA_decoupled(Distiller):
    def __init__(self, student, teacher, cfg):
        super(ReviewKD_RGA_decoupled, self).__init__(student, teacher)
        self.shapes = cfg.REVIEWKD.SHAPES
        self.out_shapes = cfg.REVIEWKD.OUT_SHAPES
        in_channels = cfg.REVIEWKD.IN_CHANNELS
        out_channels = cfg.REVIEWKD.OUT_CHANNELS
        self.ce_loss_weight = cfg.REVIEWKD.CE_WEIGHT
        self.reviewkd_loss_weight = cfg.REVIEWKD.REVIEWKD_WEIGHT
        self.warmup_epochs = cfg.REVIEWKD.WARMUP_EPOCHS
        self.stu_preact = cfg.REVIEWKD.STU_PREACT
        self.max_mid_channel = cfg.REVIEWKD.MAX_MID_CHANNEL
        self.feature_size = cfg.REVIEWKD.FEATURE_SIZE
        #
        self.use_spatial = cfg.RGA.SPATIAL
        self.use_channel = cfg.RGA.CHANNEL

        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF_RGA(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                    pow(self.feature_size[idx], 2),  #RGA_shape
                    use_spatial=self.use_spatial,
                    use_channel=self.use_channel,
                )
            )
        self.abfs = abfs[::-1]

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.abfs.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        # get features
        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x = x[::-1]
        # results = []
        spatial_result = []
        channel_result = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])  #这一个没有进真正的ABF里
        # results.append(out_features)
        #最后一层没有这个
        spatial_result.append(out_features)
        channel_result.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            _, res_features, spatial_out, channel_out = abf(features, res_features, shape, out_shape)
            spatial_result.insert(0, spatial_out)
            channel_result.insert(0, channel_out)
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        
        # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # loss_reviewkd = (
        #     self.reviewkd_loss_weight
        #     * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
        #     * hcl_loss(results, features_teacher)
        # )
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_reviewkd = (
            self.reviewkd_loss_weight
            * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
            * (hcl_loss(spatial_result, features_teacher) + hcl_loss(channel_result, features_teacher))  #loss_update
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_reviewkd,
        }
        return logits_student, losses_dict


class ABF_RGA(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse,
                 in_spatial, use_spatial=True, use_channel=True, cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(ABF_RGA, self).__init__()
        self.conv1 = nn.Sequential(
            #对student的l+1级特征进行维度转换
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            #特征融合，两级特征拼接+卷积
            self.att_conv = nn.Sequential( 
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

        #RGA_module
        self.in_channel = mid_channel * 2
        self.in_spatial = in_spatial

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        print('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

        # self.inter_channel = in_channel // cha_ratio
        # self.inter_spatial = in_spatial // spa_ratio
        self.inter_channel = max(in_channel // cha_ratio, 1)
        self.inter_spatial = max(in_spatial // spa_ratio, 1)

        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

        # Embedding functions for relation features
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # Networks for learning attention weights
        if self.use_spatial:
            num_channel_s = max(1 + self.inter_spatial, 1)  #确保至少为1
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=max(num_channel_s // down_ratio, 1), #防止为0
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(max(num_channel_s // down_ratio, 1)), #同样防止为0
                nn.ReLU(),
                nn.Conv2d(in_channels=max(num_channel_s // down_ratio, 1), out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
        if self.use_channel:
            num_channel_c = max(1 + self.inter_channel, 1) # 确保至少为1
            self.W_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_c, out_channels=max(num_channel_c // down_ratio, 1), # 防止为0
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(max(num_channel_c // down_ratio, 1)), # 同样防止为0
                nn.ReLU(),
                nn.Conv2d(in_channels=max(num_channel_c // down_ratio, 1), out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        # Embedding functions for modeling relations
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
    def forward(self, x, y=None, shape=None, out_shape=None):
        n, c, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z_spatial = z  #参与spatial_attention的提取
            z_channel = z  #参与channel_attention的提取
            spatial_att_feature = []
            channel_att_feature = []
            z_n, z_c, z_h, z_w = z.shape
            if self.use_spatial:
                # spatial attention
                theta_xs = self.theta_spatial(z_spatial)
                phi_xs = self.phi_spatial(z_spatial)
                theta_xs = theta_xs.view(z_n, self.inter_channel, -1)
                theta_xs = theta_xs.permute(0, 2, 1)
                phi_xs = phi_xs.view(z_n, self.inter_channel, -1)
                Gs = torch.matmul(theta_xs, phi_xs)
                Gs_in = Gs.permute(0, 2, 1).view(z_n, z_h * z_w, z_h, z_w)
                Gs_out = Gs.view(z_n, z_h * z_w, z_h, z_w)
                Gs_joint = torch.cat((Gs_in, Gs_out), 1)
                Gs_joint = self.gg_spatial(Gs_joint)

                g_xs = self.gx_spatial(z_spatial)
                g_xs = torch.mean(g_xs, dim=1, keepdim=True)
                ys = torch.cat((g_xs, Gs_joint), 1)

                W_ys = self.W_spatial(ys)
                z_spatial = F.sigmoid(W_ys.expand_as(z_spatial)) * z_spatial
                z_spatial = self.att_conv(z_spatial)
                spatial_att_feature = x * z_spatial[:, 0].view(n, 1, h, w) + y * z_spatial[:, 1].view(n, 1, h, w)
                

            if self.use_channel:
                # channel attention
                xc = z_channel.view(z_n, z_c, -1).permute(0, 2, 1).unsqueeze(-1)
                theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
                phi_xc = self.phi_channel(xc).squeeze(-1)
                Gc = torch.matmul(theta_xc, phi_xc)
                Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
                Gc_out = Gc.unsqueeze(-1)
                Gc_joint = torch.cat((Gc_in, Gc_out), 1)
                Gc_joint = self.gg_channel(Gc_joint)

                g_xc = self.gx_channel(xc)
                g_xc = torch.mean(g_xc, dim=1, keepdim=True)
                yc = torch.cat((g_xc, Gc_joint), 1)

                W_yc = self.W_channel(yc).transpose(1, 2)
                z_channel = F.sigmoid(W_yc) * z_channel
                z_channel = self.att_conv(z_channel)
                channel_att_feature = x * z_channel[:, 0].view(n, 1, h, w) + y * z_channel[:, 1].view(n, 1, h, w)

            if spatial_att_feature.shape[-1] != out_shape:
                spatial_att_feature = F.interpolate(spatial_att_feature, (out_shape, out_shape), mode="nearest")
            if channel_att_feature.shape[-1] != out_shape:
                channel_att_feature = F.interpolate(channel_att_feature, (out_shape, out_shape), mode="nearest")
            
            global_att_feature = spatial_att_feature + channel_att_feature  #这里看看 直接加 和 先concat再1x1conv 哪个效果好
            # if global_att_feature.shape[-1] != out_shape:
            #     global_att_feature = F.interpolate(global_att_feature, (out_shape, out_shape), mode="nearest")
            spatial_att_feature = self.conv2(spatial_att_feature)
            channel_att_feature = self.conv2(channel_att_feature)
            #if fusion, return below
            return y, global_att_feature, spatial_att_feature, channel_att_feature

        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)

        #x传入上一级，y和teacher进行计算
        #那么我现在改了之后，y还是y，但是作loss的就变成spatial_att_feature和channel_att_feature了，这两个也要传出来
        #虽然y要传出来，但不参与loss的计算
        # if not fusion, return below
        return y, x