from .baseLayer import *
from compressai.models import deeplab
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2 import model_zoo
import pickle


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64



class FasterRCNN_Coding(CompressionModel):
    def __init__(self,
                 pretrain_img_size=256,
                 patch_size=2,
                 in_chans=3,
                 embed_dim=48,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=4,
                 num_slices=2,
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

        cfg = get_cfg()  # get default cfg
        cfg.merge_from_file("/home/exx/Documents/Tianma/ICM/config/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = '/home/exx/Documents/Tianma/ICM/save_model/R50-FPN_x3.pkl'

        self.task_net = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
        checkpoint = OrderedDict()
        with open(cfg.MODEL.WEIGHTS, 'rb') as f:
            FPN_ckpt = pickle.load(f)
            for k, v in FPN_ckpt['model'].items():
                if 'backbone' in k:
                    checkpoint['.'.join(k.split('.')[1:])] = torch.from_numpy(v)
        self.task_net.load_state_dict(checkpoint, strict=True)
        self.task_net = self.task_net.to('cuda')
        for k, p in self.task_net.named_parameters():
            p.requires_grad = False


        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        self.num_slices = num_slices
        self.max_support_slices = 4
        self.support_num = 8

        N = 192
        M = 384

        self.g_a = mainCNNencoder(N,M)
        self.g_s = mainCNNdecoder(N, M)


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

        self.gaussian_conditional = GaussianConditional(None)


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
        # compressH, Teacher_output_features, Teacher_classification, Teacher_regression, Teacher_anchors = None, None, None, None, None
        inputIMGs = x.clone()
        with torch.no_grad():
            teacher_out = self.task_net(x)

        # print(inputIMGs.shape)

        y = self.g_a(inputIMGs)

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
            # print(y_zigzag[:,slice_index,:,:,:].shape)
            # print(scale.shape)
            # print(mu.shape)
            _, y_slice_likelihood = self.gaussian_conditional(y_zigzag[:,slice_index,:,:,:], scale, mu)

            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_zigzag[:,slice_index,:,:,:] - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms2[slice_index](lrp_support)

            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp


            y_hat_slices.append(y_hat_slice)

        # y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = torch.cat(y_hat_slices, dim=1).view(-1,self.num_slices* num_H* num_W,384 //self.num_slices,(H//number),(W//number))
        y_hat = self.ZigzagReverse(y_hat,self.num_slices, number, number)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        x_hat = self.g_s(y_hat)

        student_out = self.task_net(x_hat)


        return {
            "decompressedImage":x_hat,
            # "compressH": compressH,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "Student_output_features": student_out,
            "Teacher_output_features": teacher_out,
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