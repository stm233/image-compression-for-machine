from .baseLayer import *
from compressai.models import deeplab

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64



class ConditionalResidualCoding3(CompressionModel):
    def __init__(self,
                 pretrain_img_size=256,
                 patch_size=2,
                 in_chans=3,
                 embed_dim=48,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=4,
                 num_slices=6,
                 Mask_win_size=8,
                 num_sliding = 4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.num_slices = num_slices
        self.max_support_slices = 12
        self.support_num = 24
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        # self.layers = mainEncoder(self.num_layers,embed_dim,depths,
        #                           num_heads,window_size,mlp_ratio,qkv_bias,
        #                           qk_scale,drop_rate,attn_drop_rate,dpr,
        #                           norm_layer,use_checkpoint)

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        # self.syn_layers = mainDecoder(self.num_layers, embed_dim, depths,
        #                               num_heads, window_size, mlp_ratio,
        #                               qkv_bias, qk_scale, drop_rate,
        #                                 attn_drop_rate, dpr, norm_layer,
        #                               use_checkpoint)


        # LRP_depths = [2,6]
        # self.num_LRP = len(LRP_depths)
        # self.LRP_Swin2 = nn.ModuleList()
        # for i in range(num_slices*4):
        #     LRP_layer = nn.ModuleList()
        #     for i_layer in range(self.num_LRP):
        #         layer = BasicLayer(
        #             dim=int(384 //self.num_slices),
        #             depth=LRP_depths[i_layer],
        #             num_heads=4,
        #             window_size=8,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,
        #             drop=drop_rate,
        #             attn_drop=attn_drop_rate,
        #             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #             norm_layer=norm_layer,
        #             downsample= None,
        #             use_checkpoint=use_checkpoint,
        #             inverse=True)
        #         LRP_layer.append(layer)
        #
        #     self.LRP_Swin2.append(LRP_layer)

        self.end_conv = nn.Sequential(nn.Conv2d(embed_dim, embed_dim * patch_size ** 2, kernel_size=5, stride=1, padding=2),
                                      nn.PixelShuffle(patch_size),
                                      nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
                                      )

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        self.num_features = num_features
        N = 192
        M = 384

        self.g_a = mainCNNencoder(N,M)
        self.g_s = mainCNNdecoder(N, M)
        self.g_s1 = mainCNNdecoderPart1(N,M)
        self.g_s2 = mainCNNdecoderPart2(N) # output channel = 3

        self.human_g_enc2 = mainCNNcontextScale1(N,M) # output channel = 3
        self.human_g_enc3 = mainCNNcontextScale2(N,M) # output channel = N
        self.human_g_enc4 = mainCNNcontextScale1(N, M)  # output channel = 3
        self.human_g_enc5 = mainCNNcontextScale2(N, M)  # output channel = N

        self.seg_g_enc2 = mainCNNcontextScale1(N, M)
        self.seg_g_enc3 = mainCNNcontextScale2(N, M)


        self.h_a = hyperEncoder()

        self.h_mean_s = hyperMean()
        self.h_scale_s = hyperMean()
        number = 2
        self.cc_mean_transforms2 = hyperContextMean(self.support_num, self.num_slices,
                                                    self.max_support_slices, number)
        self.cc_scale_transforms2 = hyperContextMean(self.support_num, self.num_slices,
                                                    self.max_support_slices, number)
        self.lrp_transforms2 = hyperContextLRP(self.support_num,self.num_slices,self.max_support_slices, number)
        self.entropy_bottleneck = EntropyBottleneck(embed_dim * 4)
        self.entropy_bottleneck_human = EntropyBottleneck(embed_dim * 4)
        self.entropy_bottleneck_seg = EntropyBottleneck(embed_dim * 4)
        self.gaussian_conditional = GaussianConditional(None)
        self.gaussian_conditional_human = GaussianConditional(None)
        self.gaussian_conditional_seg = GaussianConditional(None)
        self._freeze_stages()

        # self.teacherNet = model.resnet50(num_classes=80, pretrained=False)
        # self.studentNet = model.studentresnet50(num_classes=80, pretrained=False)
        #
        # self.teacherNet.training = True
        # self.studentNet.training = True
        #
        # self.student_seg_Net = deeplab.modeling.__dict__['deeplabv3_resnet50'](num_classes=21, output_stride=16)

        self.seg_h_mean_s = hyperMean()
        self.seg_h_scale_s = hyperMean()
        number = 2
        self.seg_cc_mean_transforms2 = hyperContextMean(self.support_num, self.num_slices,
                                                    self.max_support_slices, number)
        self.seg_cc_scale_transforms2 = hyperContextMean(self.support_num, self.num_slices,
                                                     self.max_support_slices, number)
        self.seg_lrp_transforms2 = hyperContextLRP(self.support_num,self.num_slices,self.max_support_slices, number)
        self.seg_g_s = mainCNNdecoder(N, M)

        self.human_g_a1_2 = nn.Sequential(
            conv(9, N, kernel_size=3, stride=2),
            nn.GELU(),
            conv(N, N, kernel_size=3, stride=2),
        )
        self.seg_g_a1 = nn.Sequential(
            conv(6, N, kernel_size=3, stride=2),
            nn.GELU(),
            conv(N, N, kernel_size=3, stride=2),
        )

        self.human_g_a2_2 = nn.Sequential(
            conv(N + N + N, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, M, kernel_size=5, stride=2),
            nn.GELU(),
            # Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.seg_g_a2 = nn.Sequential(
            conv(N + N, N, kernel_size=5, stride=2),
            nn.GELU(),
            conv(N, M, kernel_size=5, stride=2),
            nn.GELU(),
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )

        self.human_g_s1_2 = nn.Sequential(
            # Win_noShift_Attention(dim=M*3, num_heads=8, window_size=4, shift_size=2),
            # nn.GELU(),
            deconv(M*3, N, kernel_size=3, stride=2),
            nn.GELU(),
            deconv(N, N, kernel_size=3, stride=2),
            # nn.GELU(),
            # deconv(N, N, kernel_size=5, stride=2),
            # nn.GELU(),
            # deconv(N, 3, kernel_size=5, stride=2),
        )

        self.human_g_s2_2 = nn.Sequential(
            deconv(N*3, N, kernel_size=3, stride=2),
            nn.GELU(),
            conv(N , N, kernel_size=3, stride=1),
            nn.GELU(),
            deconv(N, 3, kernel_size=3, stride=2),
        )

        self.human_h_a = hyperEncoder()
        self.seg_h_a = hyperEncoder()

        self.generate_mask_scale1 = nn.Sequential(
            conv3x3(6, 12),
            nn.GELU(),
            conv3x3(12, 12),
            nn.GELU(),
            conv3x3(12, 9),
            nn.Softmax(dim=1),
        )

        self.generate_mask_scale2 = nn.Sequential(
            conv3x3(2*N, 4*N),
            nn.GELU(),
            conv3x3(4*N, 4*N),
            nn.GELU(),
            conv3x3(4*N, 3*N),
            nn.Softmax(dim=1),
        )



        self.human_h_mean_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            # subpel_conv3x3(240, 288, 2),
            deconv(240, 288, kernel_size=3, stride=2),
            nn.GELU(),
            # conv3x3(288, 336),
            # nn.GELU(),
            # subpel_conv3x3(288, 384, 2),
            deconv(288, 384, kernel_size=3, stride=2),
            # nn.GELU(),
            # conv3x3(384, 384),
            # nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),

        )
        self.human_h_scale_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            deconv(240, 288, kernel_size=3, stride=2),
            nn.GELU(),
            # conv3x3(288, 336),
            # nn.GELU(),
            deconv(288, 384, kernel_size=3, stride=2),
            # nn.GELU(),
            # conv3x3(384, 384),
            # nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
        )
        self.human_context_decoder = nn.Sequential(
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
        )

        self.human_context_decoder2 = nn.Sequential(
            conv(384, 192, stride=1, kernel_size=3),
            nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            deconv(192, 192, kernel_size=3, stride=2),
            # subpel_conv3x3(384, 192, 2),
            nn.GELU(),
            # subpel_conv3x3(192, 192, 2),
            deconv(192, 192, kernel_size=3, stride=2),
        )
        self.human_context_decoder3 = nn.Sequential(
            conv(384, 384, stride=1, kernel_size=3),
            nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            conv(384, 384, stride=1, kernel_size=3),
        )

        self.human_context_decoder4 = nn.Sequential(
            conv(384, 192, stride=1, kernel_size=3),
            nn.GELU(),
            # conv(384, 384, stride=1, kernel_size=3),
            # nn.GELU(),
            deconv(192, 192, kernel_size=3, stride=2),
            # subpel_conv3x3(384, 192, 2),
            nn.GELU(),
            # subpel_conv3x3(192, 192, 2),
            deconv(192, 192, kernel_size=3, stride=2),
        )

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def ZigzagSplits(self,inputs,num_slices):
        B,C,H,W = inputs.shape
        number = 2
        # window_size = 8
        pad_l = pad_t = 0
        pad_r = ((H//number) - H % (H//number)) % (H//number)
        pad_b = ((W//number) - W % (W//number)) % (W//number)
        x = F.pad(inputs, (pad_t, pad_b, pad_l, pad_r))
        _,_,Hpad,Wpad = x.shape
        # print(x.shape)

        x_slices = x.view(B,num_slices,C//num_slices,
                          number,H//number,
                          number,W//number)

        outOrders = []
        embedding_len = (H//number)*(W//number)*C//num_slices

        for i in range(max(number,number)):
            C_index = 0
            H_index = W_index = 0
            for j in range(num_slices*min((i+1),number)*min((i+1),number)):

                if max(H_index,W_index) < i and i > 0:
                    # print('N',C_index,H_index,W_index)
                    if C_index + 2 > num_slices : # or C_index +1> i
                        C_index = 0
                        if H_index + 2 > number or H_index+1 > i:
                            W_index = W_index + 1
                            H_index = 0
                        else:
                            H_index = H_index + 1

                    else:
                        C_index = C_index + 1
                    continue

                # print('Y',C_index,H_index,W_index,'i=',i)
                outOrders.append(x_slices[:,C_index,:,H_index,:,W_index,:].contiguous().unsqueeze(1))
                if C_index + 2 > num_slices : # or C_index +1 > i
                    C_index = 0
                    if H_index + 2 > number or H_index+1 > i:
                        W_index = W_index + 1
                        H_index = 0
                    else:
                        H_index = H_index + 1

                else:
                    C_index = C_index + 1

        zigzag = torch.cat(outOrders, 1).to(inputs.device)

        return zigzag, number, number


    def ZigzagReverse(self,inputs,num_slices,num_H,num_W):
        # inputs : list
        B,N,C,H,W = inputs.shape
        out_C = C*num_slices
        out_H = H*num_H
        out_W = W*num_W

        output = torch.zeros(B,out_C,out_H,out_W, device = inputs.device)
        output = output.view(B,num_slices,C,
                               num_H     ,H,
                               num_W     ,W)
        inputs_index = 0

        for i in range(max(num_H, num_W)):
            C_index = 0
            H_index = W_index = 0
            for j in range(num_slices*min((i+1),num_H)*min((i+1),num_W)):

                if max(H_index,W_index) < i and i > 0:
                    # print('N',C_index,H_index,W_index)
                    if C_index + 2 > num_slices: # or C_index +1> i:
                        C_index = 0
                        if H_index + 2 > num_H or H_index+1 > i:
                            W_index = W_index + 1
                            H_index = 0
                        else:
                            H_index = H_index + 1

                    else:
                        C_index = C_index + 1
                    continue
                # print('Y',C_index,H_index,W_index,'i=',i)
                output[:,C_index,:,H_index,:,W_index,:] = inputs[:,inputs_index]
                inputs_index = inputs_index + 1
                if C_index + 2 > num_slices: # or C_index +1 > i:
                    C_index = 0
                    if H_index + 2 > num_H or H_index+1 > i:
                        W_index = W_index + 1
                        H_index = 0
                    else:
                        H_index = H_index + 1

                else:
                    C_index = C_index + 1

        output = output.view(B,out_C,out_H,out_W).contiguous()
        return output

    def forward(self, x):
        """Forward function."""
        # compressH, Teacher_output_features, Teacher_classification, Teacher_regression, Teacher_anchors = self.teacherNet(x)
        compressH, Teacher_output_features, Teacher_classification, Teacher_regression, Teacher_anchors = None, None, None, None, None
        inputIMGs = x.clone()
        # print(inputIMGs.shape)

        y = self.g_a(x)

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        B,C,H,W = latent_scales.shape
        number = 2

        y_zigzag, num_H, num_W = self.ZigzagSplits(y, self.num_slices)
        scales_zigzag, _ , _    = self.ZigzagSplits(latent_scales,self.num_slices)
        means_zigzag, _ , _   = self.ZigzagSplits(latent_means,self.num_slices)

        # y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index in range(self.num_slices* num_H* num_W):
            # support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            support_slices = (y_hat_slices if self.max_support_slices > slice_index else y_hat_slices[slice_index-self.max_support_slices:])

            support_num = self.support_num
            # meanInput = ([means_zigzag[:,slice_index:slice_index+self.max_support_slices,:,:,:]] if slice_index > slice_index else y_hat_slices[slice_index-self.max_support_slices:])
            if slice_index+support_num > self.num_slices* num_H* num_W:
                meanInput = means_zigzag[:,-support_num:,:,:,:].view(-1,384*support_num//self.num_slices,(H//number),(W//number))
            else:
                meanInput = means_zigzag[:,slice_index:slice_index+support_num,:,:,:].view(-1,384*support_num//self.num_slices,(H//number),(W//number))
            mean_support = torch.cat([meanInput] + support_slices, dim=1)
            # print(slice_index,'support_slices',len(support_slices),mean_support.shape)
            mu = self.cc_mean_transforms2[slice_index](mean_support)
            # mu = mu[:, :, :y_shape[0], :y_shape[1]]

            if slice_index+support_num > self.num_slices* num_H* num_W:
                scaleInput = scales_zigzag[:,-support_num:,:,:,:].view(-1,384*support_num//self.num_slices,(H//number),(W//number))
            else:
                scaleInput = scales_zigzag[:,slice_index:slice_index+support_num,:,:,:].view(-1,384*support_num//self.num_slices,(H//number),(W//number))
            # scaleInput = scales_zigzag.view(-1,384*number*number,(H//number),(W//number))
            scale_support = torch.cat([scaleInput] + support_slices, dim=1)
            scale = self.cc_scale_transforms2[slice_index](scale_support)
            # scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # mu1 = mu.clone()
            # mu1 = mu1.permute(0, 2, 3, 1).contiguous().view(-1, (H//number)*(W//number), 384 //self.num_slices)
            # for i in range(self.num_mu):
            #     mu_layer = self.mu_Swin2[slice_index][i]#.to(y.device)
            #     mu1, _, _ = mu_layer(mu1, (H//number), (W//number))
            #
            # mu = mu + mu1.view(-1, (H//number), (W//number), 384 //self.num_slices).permute(0, 3, 1, 2).contiguous()

            # scale1 = scale.clone()
            # scale1 = scale1.permute(0, 2, 3, 1).contiguous().view(-1,  (H//number)*(W//number), 384 //self.num_slices)
            # for i in range(self.num_sigma):
            #     layer = self.sigma_Swin2[slice_index][i]#.to(y.device)
            #     scale1, _, _ = layer(scale1,(H//number), (W//number))
            #
            # scale = scale + scale1.view(-1, (H//number), (W//number), 384 //self.num_slices).permute(0, 3, 1, 2).contiguous()

            _, y_slice_likelihood = self.gaussian_conditional(y_zigzag[:,slice_index,:,:,:], scale, mu)

            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_zigzag[:,slice_index,:,:,:] - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms2[slice_index](lrp_support)

            # lrp1 = lrp.clone()
            # lrp1 = lrp1.permute(0, 2, 3, 1).contiguous().view(-1, (H//number)*(W//number), 384 //self.num_slices)
            # for i in range(self.num_LRP):
            #     layer = self.LRP_Swin2[slice_index][i]#.to(y.device)
            #     lrp1, _, _ = layer(lrp1, (H//number), (W//number))
            #
            # lrp = lrp + lrp1.view(-1, (H//number), (W//number), 384 //self.num_slices).permute(0, 3, 1, 2).contiguous()
            #
            #
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp


            y_hat_slices.append(y_hat_slice)

        # y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = torch.cat(y_hat_slices, dim=1).view(-1,self.num_slices* num_H* num_W,384 //self.num_slices,(H//number),(W//number))
        y_hat = self.ZigzagReverse(y_hat,self.num_slices, number, number)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        x_hat = self.g_s(y_hat)
        # h_hat1 = self.g_s1(y_hat)
        # promot_h_hat = self.promot_g_s(y_hat)
        # h_hat = promot_h_hat + h_hat
        # decompressImage = self.g_s2(h_hat1)

        # decompressH = self.end_conv(y_hat.view(-1, Wh, Ww, self.embed_dim).permute(0, 3, 1, 2).contiguous())

        # Student_compressH, Student_output_features, Student_classification, Student_regression, Student_anchors, scores, labels, boxes = self.studentNet(h_hat,decompressImage)
        # y_likelihoods, z_likelihoods, Student_classification, Student_regression = None,None,None,None
        Student_compressH, Student_output_features, Student_classification, Student_regression, Student_anchors, scores, labels, boxes = None,None,None,None,None,None,None,None
        # Student_compressH, Student_output_features, Student_classification, Student_regression, Student_anchors, scores, labels, boxes = self.studentNet(x)



        # -----------------------------------------------------------------------
        # segmentation -----------------------------------------------------------------------
        seg_decompressImage = self.seg_g_enc2(y_hat)
        seg_conditionalScale = self.seg_g_enc3(y_hat)
        seg_support = torch.cat([inputIMGs, seg_decompressImage], dim=1)
        seg_y_1 = self.seg_g_a1(seg_support)

        seg_support2 = torch.cat([seg_y_1, seg_conditionalScale], dim=1)
        seg_y = self.seg_g_a2(seg_support2)
        # print(seg_y.shape, y.shape)
        seg_z = self.seg_h_a(seg_y)

        _, seg_z_likelihoods = self.entropy_bottleneck_seg(z)
        seg_z_offset = self.entropy_bottleneck_seg._get_medians()
        seg_z_tmp = seg_z - seg_z_offset
        seg_z_hat = ste_round(seg_z_tmp) + seg_z_offset

        seg_latent_scales = self.seg_h_scale_s(seg_z_hat)
        seg_latent_means = self.seg_h_mean_s(seg_z_hat)

        B, C, H, W = seg_latent_scales.shape
        number = 2

        seg_y_zigzag, num_H, num_W = self.ZigzagSplits(seg_y, self.num_slices)
        seg_scales_zigzag, _, _ = self.ZigzagSplits(seg_latent_scales, self.num_slices)
        seg_means_zigzag, _, _ = self.ZigzagSplits(seg_latent_means, self.num_slices)

        # y_slices = y.chunk(self.num_slices, 1)
        seg_y_hat_slices = []
        seg_y_likelihood = []

        for slice_index in range(self.num_slices * num_H * num_W):
            # support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            seg_support_slices = (seg_y_hat_slices if self.max_support_slices > slice_index else seg_y_hat_slices[
                                                                                         slice_index - self.max_support_slices:])

            support_num = self.support_num
            # meanInput = ([means_zigzag[:,slice_index:slice_index+self.max_support_slices,:,:,:]] if slice_index > slice_index else y_hat_slices[slice_index-self.max_support_slices:])
            if slice_index + support_num > self.num_slices * num_H * num_W:
                seg_meanInput = seg_means_zigzag[:, -support_num:, :, :, :].view(-1, 384 * support_num // self.num_slices,
                                                                         (H // number), (W // number))
            else:
                seg_meanInput = seg_means_zigzag[:, slice_index:slice_index + support_num, :, :, :].view(-1,
                                                                                                 384 * support_num // self.num_slices,
                                                                                                 (H // number),
                                                                                                 (W // number))
            seg_mean_support = torch.cat([seg_meanInput] + seg_support_slices, dim=1)
            # print(slice_index,'support_slices',len(support_slices),mean_support.shape)
            mu = self.seg_cc_mean_transforms2[slice_index](seg_mean_support)
            # mu = mu[:, :, :y_shape[0], :y_shape[1]]

            if slice_index + support_num > self.num_slices * num_H * num_W:
                seg_scaleInput = seg_scales_zigzag[:, -support_num:, :, :, :].view(-1, 384 * support_num // self.num_slices,
                                                                           (H // number), (W // number))
            else:
                seg_scaleInput = seg_scales_zigzag[:, slice_index:slice_index + support_num, :, :, :].view(-1,
                                                                                                   384 * support_num // self.num_slices,
                                                                                                   (H // number),
                                                                                                   (W // number))
            # scaleInput = scales_zigzag.view(-1,384*number*number,(H//number),(W//number))
            seg_scale_support = torch.cat([seg_scaleInput] + seg_support_slices, dim=1)
            scale = self.seg_cc_scale_transforms2[slice_index](seg_scale_support)

            _, seg_y_slice_likelihood = self.gaussian_conditional_seg(seg_y_zigzag[:, slice_index, :, :, :], scale, mu)

            seg_y_likelihood.append(seg_y_slice_likelihood)
            seg_y_hat_slice = ste_round(seg_y_zigzag[:, slice_index, :, :, :] - mu) + mu

            seg_lrp_support = torch.cat([seg_mean_support, seg_y_hat_slice], dim=1)
            lrp = self.seg_lrp_transforms2[slice_index](seg_lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            seg_y_hat_slice += lrp



            seg_y_hat_slices.append(seg_y_hat_slice)

        seg_y_hat = torch.cat(seg_y_hat_slices, dim=1).view(-1, self.num_slices * num_H * num_W, 384 // self.num_slices,
                                                    (H // number), (W // number))
        seg_y_hat = self.ZigzagReverse(seg_y_hat, self.num_slices, number, number)
        seg_y_likelihood = torch.cat(seg_y_likelihood, dim=1)

        seg_x_hat = self.seg_g_s(seg_y_hat)
        # Student_seg_output = self.student_seg_Net(seg_x_hat)
        Student_seg_output = None

        #-----------------------------------------------------------------------------
        decompressImage2 = self.human_g_enc2(y_hat)
        conditionalScale2 = self.human_g_enc3(y_hat)
        decompressImage3 = self.human_g_enc4(seg_y_hat)
        conditionalScale4 = self.human_g_enc5(seg_y_hat)

        mask_scale1 = self.generate_mask_scale1(torch.cat([decompressImage2, decompressImage3], dim=1))
        mask_obj1 = mask_scale1[:,0:3,:,:]
        mask_seg1 = mask_scale1[:, 3:6, :, :]
        # print(mask_obj1.shape,mask_obj2.shape)
        # print(torch.sum(mask_obj1))

        # decompressImage2 = decompressImage2 + decompressImage3
        # conditionalScale2 = conditionalScale2 + conditionalScale4
        # print(decompressImage2.shape,conditionalScale2.shape,inputIMGs.shape)

        residual1 = inputIMGs - mask_obj1 * decompressImage2 - mask_seg1 * decompressImage3
        human_support = torch.cat([residual1, decompressImage2, decompressImage3], dim=1)
        human_y_1 = self.human_g_a1_2(human_support)

        mask_scale2= self.generate_mask_scale2(torch.cat([conditionalScale2, conditionalScale4], dim=1))
        mask_obj2 = mask_scale2[:, 0:192, :, :]
        mask_seg2 = mask_scale2[:, 192:384, :, :]
        residual2 = human_y_1 - mask_obj2 * conditionalScale2 - mask_seg2 * conditionalScale4
        human_support2 = torch.cat([residual2, conditionalScale2, conditionalScale4], dim=1)

        human_y = self.human_g_a2_2(human_support2)
        human_z = self.human_h_a(human_y)
        # promot_z = self.promot_h_a(y)
        # z = z + promot_z
        _, human_z_likelihoods = self.entropy_bottleneck_human(z)
        human_z_offset = self.entropy_bottleneck_human._get_medians()
        human_z_tmp = human_z - human_z_offset
        human_z_hat = ste_round(human_z_tmp) + human_z_offset

        human_latent_scales = self.human_h_scale_s(human_z_hat)
        human_latent_means = self.human_h_mean_s(human_z_hat)

        _, human_y_likelihood = self.gaussian_conditional_human(human_y, human_latent_scales, human_latent_means)

        # human_y_likelihood.append(y_slice_likelihood)
        human_y_hat= ste_round(human_y - human_latent_means) + human_latent_means

        context = self.human_context_decoder(y_hat)
        context2 = self.human_context_decoder2(y_hat)

        context3 = self.human_context_decoder3(seg_y_hat)
        context4 = self.human_context_decoder4(seg_y_hat)
        # context = context + context3
        # context2 = context4 + context2

        decoder_support = torch.cat([human_y_hat, context,context3], dim=1)

        human_deimage1 = self.human_g_s1_2(decoder_support)
        human_deimage1 = human_deimage1 + mask_obj2 * conditionalScale2 + mask_seg2 * conditionalScale4


        decoder_support2 = torch.cat([human_deimage1, context2,context4], dim=1)
        human_deimage = self.human_g_s2_2(decoder_support2)
        human_deimage = human_deimage + mask_obj1 * decompressImage2 + mask_seg1 * decompressImage3

        return {
            "decompressedImage":human_deimage,
            "compressH": compressH,
            "decompressH": Student_compressH,
            "likelihoods": {"y": human_y_likelihood, "z": human_z_likelihoods},
            "Student_output_features": Student_seg_output,
            "Teacher_output_features": Teacher_output_features,
            "Student_classification": Student_classification,
            "Student_regression": Student_regression,
            "Student_anchors": Student_anchors,
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net



if __name__ == '__main__':
    # x = [torch.rand(3, 640, 640), torch.rand(3, 640, 640)]
    x = torch.rand(1, 3, 1280, 1280)
    teacher_model = TeacherModel()

    compressH,output_features,classification, regression, anchors = teacher_model(x)
    test_retina = teacher_model.model.eval(x)
