from monai.networks.nets import UNet as MONAI_UNet, UNETR
from .blocks import PlainBlock, ResidualBlock
from .unet import MultiEncoderUNet   # keep only if you still need your custom MultiEncoderUNet
from monai.networks.nets import UNet as MONAI_UNet, UNETR
import monai
print("MONAI version:", monai.__version__)
from torchsummary import summary

block_dict = {
    'plain': PlainBlock,
    'res': ResidualBlock
}

def get_unet(args):
    if args.unet_arch == 'unet':
        return MONAI_UNet(
            spatial_dims=3,
            in_channels=args.input_channels,
            out_channels=args.num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=args.norm,
            dropout=args.dropout_prob,
        )
    elif args.unet_arch == 'multiencoder_unet':
        return MultiEncoderUNet(
            input_channels=args.input_channels,
            output_classes=args.num_classes,
            channels_list=args.channels_list,
            deep_supervision=args.deep_supervision,
            ds_layer=args.ds_layer,
            kernel_size=args.kernel_size,
            dropout_prob=args.dropout_prob,
            norm_key=args.norm,
            block=block_dict[args.block],
        )
    elif args.unet_arch == 'unetr':
        return UNETR(
            spatial_dims=3,
            in_channels=args.input_channels,
            out_channels=args.num_classes,
            img_size=(args.patch_size, args.patch_size, args.patch_size),
            norm_name=args.norm,
            dropout_rate=args.dropout_prob,
        )
    elif args.unet_arch == 'unet2':
        return MONAI_UNet(
            spatial_dims=3,
            in_channels=args.input_channels,  # usually 4 for BraTS
            out_channels=args.num_classes,  # usually 3 for BraTS
            channels=(32, 64, 128, 256, 320),  # nnU-Net standard
            strides=(2, 2, 2, 2),
            num_res_units=2,  # 2 convs per stage
            kernel_size=3,  # nnU-Net default
            act=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
            norm="instance",
            dropout=0.0,
        )

    else:
        raise NotImplementedError(args.unet_arch + " is not implemented.")



# # from .blocks import PlainBlock, ResidualBlock
# # from .unet import UNet, MultiEncoderUNet
# # from monai.networks.nets import SwinUNETR
# # from torchsummary import summary
# #
# # block_dict = {
# #     'plain': PlainBlock,
# #     'res': ResidualBlock
# # }
# #
# #
# # def get_unet(args):
# #     kwargs = {
# #         "input_channels"   : args.input_channels,
# #         "output_classes"   : args.num_classes,
# #         "channels_list"    : args.channels_list,
# #         "deep_supervision" : args.deep_supervision,
# #         "ds_layer"         : args.ds_layer,
# #         "kernel_size"      : args.kernel_size,
# #         "dropout_prob"     : args.dropout_prob,
# #         "norm_key"         : args.norm,
# #         "block"            : block_dict[args.block],
# #     }
# #
# #     if args.unet_arch == 'unet':
# #         return UNet(**kwargs)
# #
# #     elif args.unet_arch == 'multiencoder_unet':
# #         return MultiEncoderUNet(**kwargs)
# #     elif args.unet_arch == 'SwinUNETR':
# #         return SwinUNETR(
# #             in_channels=args.input_channels,
# #             out_channels=args.num_classes,
# #             img_size=(args.patch_size, args.patch_size, args.patch_size),
# #
# #
# #             feature_size=24,
# #             drop_rate=args.dropout_prob,
# #             norm_name=args.norm,
# #             use_checkpoint=False,
# #             spatial_dims=3,
# #             downsample="merging",
# #             use_v2=False,
# #         )
# #     else:
# #         raise NotImplementedError(args.unet_arch + " is not implemented.")
# #
#
# # from .blocks import PlainBlock, ResidualBlock
# # from .unet import UNet, MultiEncoderUNet
# # from monai.networks.nets import ViT
# # from torchsummary import summary
# #
# # block_dict = {
# #     'plain': PlainBlock,
# #     'res': ResidualBlock
# # }
# #
# #
# # def get_unet(args):
# #     kwargs = {
# #         "input_channels"   : args.input_channels,
# #         "output_classes"   : args.num_classes,
# #         "channels_list"    : args.channels_list,
# #         "deep_supervision" : args.deep_supervision,
# #         "ds_layer"         : args.ds_layer,
# #         "kernel_size"      : args.kernel_size,
# #         "dropout_prob"     : args.dropout_prob,
# #         "norm_key"         : args.norm,
# #         "block"            : block_dict[args.block],
# #     }
# #
# #     if args.unet_arch == 'unet':
# #         return UNet(**kwargs)
# #
# #     elif args.unet_arch == 'multiencoder_unet':
# #         return MultiEncoderUNet(**kwargs)
# #     elif args.unet_arch == 'VIT':
# #         return ViT(
# #             in_channels=args.input_channels,
# #             img_size=(args.patch_size, args.patch_size, args.patch_size),
# #             patch_size=16,
# #             hidden_size=768,
# #             mlp_dim=3072,
# #             num_heads=12,
# #             pos_embed='perceptron',
# #             classification=False,
# #             dropout_rate=args.dropout_prob,
# #             spatial_dims=3
# #         )
# #     else:
# #         raise NotImplementedError(args.unet_arch + " is not implemented.")
#
# #
# from .blocks import PlainBlock, ResidualBlock
# from .unet import UNet, MultiEncoderUNet
# from monai.networks.nets import BasicUNet
# from torchsummary import summary
#
# block_dict = {
#     'plain': PlainBlock,
#     'res': ResidualBlock
# }
#
#
# def get_unet(args):
#     kwargs = {
#         "input_channels"   : args.input_channels,
#         "output_classes"   : args.num_classes,
#         "channels_list"    : args.channels_list,
#         "deep_supervision" : args.deep_supervision,
#         "ds_layer"         : args.ds_layer,
#         "kernel_size"      : args.kernel_size,
#         "dropout_prob"     : args.dropout_prob,
#         "norm_key"         : args.norm,
#         "block"            : block_dict[args.block],
#     }
#
#     if args.unet_arch == 'unet':
#         return UNet(**kwargs)
#
#     elif args.unet_arch == 'multiencoder_unet':
#         return MultiEncoderUNet(**kwargs)
#     elif args.unet_arch == 'BasicUNet':
#         return BasicUNet(
#             spatial_dims=3,  # Keep spatial dims as 3 for 3D images
#             in_channels=args.input_channels,
#             out_channels=args.num_classes,
#             features=args.channels_list,  # Features are passed from channels_list
#             act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),  # Activation function
#             norm=("instance", {"affine": True}),  # Normalization
#             bias=True,  # Keep bias as True unless specified otherwise
#             dropout=args.dropout_prob,  # Dropout probability
#             upsample="deconv",  # Use deconvolution for upsampling
#         )
#     else:
#         raise NotImplementedError(args.unet_arch + " is not implemented.")

#
# #
# from .blocks import PlainBlock, ResidualBlock
# from .unet import UNet, MultiEncoderUNet
# from monai.networks.nets import SwinUNETR
# from torchsummary import summary
#
# block_dict = {
#     'plain': PlainBlock,
#     'res': ResidualBlock
# }
#
#
# def get_unet(args):
#     kwargs = {
#         "input_channels"   : args.input_channels,
#         "output_classes"   : args.num_classes,
#         "channels_list"    : args.channels_list,
#         "deep_supervision" : args.deep_supervision,
#         "ds_layer"         : args.ds_layer,
#         "kernel_size"      : args.kernel_size,
#         "dropout_prob"     : args.dropout_prob,
#         "norm_key"         : args.norm,
#         "block"            : block_dict[args.block],
#     }
#
#     if args.unet_arch == 'unet':
#         return UNet(**kwargs)
#
#     elif args.unet_arch == 'multiencoder_unet':
#         return MultiEncoderUNet(**kwargs)
#     elif args.unet_arch == 'SwinUNETR':
#         return SwinUNETR(
#             in_channels=args.input_channels,
#             out_channels=args.num_classes,
#             img_size=(args.patch_size, args.patch_size, args.patch_size),
#
#
#             feature_size=12,
#             drop_rate=args.dropout_prob,
#             norm_name=args.norm,
#             use_checkpoint=False,
#             spatial_dims=3,
#             downsample="merging",
#             use_v2=False,
#         )
#     else:
#         raise NotImplementedError(args.unet_arch + " is not implemented.")
#
# # from .blocks import PlainBlock, ResidualBlock
# # from .unet import UNet, MultiEncoderUNet
# # from monai.networks.nets import SegResNet, DynUNet
# # from torchsummary import summary
# #
# # block_dict = {
# #     'plain': PlainBlock,
# #     'res': ResidualBlock
# # }
# #
# # def get_unet(args):
# #     kwargs = {
# #         "input_channels": args.input_channels,
# #         "output_classes": args.num_classes,
# #         "channels_list": args.channels_list,
# #         "deep_supervision": args.deep_supervision,
# #         "ds_layer": args.ds_layer,
# #         "kernel_size": args.kernel_size,
# #         "dropout_prob": args.dropout_prob,
# #         "norm_key": args.norm,
# #         "block": block_dict[args.block],
# #     }
# #
# #     if args.unet_arch == 'unet':
# #         return UNet(**kwargs)
# #
# #     elif args.unet_arch == 'multiencoder_unet':
# #         return MultiEncoderUNet(**kwargs)
# #
# #     elif args.unet_arch == 'segresnet':
# #         return SegResNet(
# #             in_channels=args.input_channels,
# #             out_channels=args.num_classes,
# #             init_filters=args.init_filters if hasattr(args, "init_filters") else 32,
# #             dropout_prob=args.dropout_prob,
# #             norm=args.norm,
# #         )
# #
# #     elif args.unet_arch == 'DynUNet':
# #         return DynUNet(
# #             spatial_dims=getattr(args, "spatial_dims", 3),  # default to 3D        # 2D or 3D
# #             in_channels=args.input_channels,
# #             out_channels=args.num_classes,
# #             kernel_size=[[3, 3, 3]] * len(args.channels_list),          # e.g. [[3,3,3], [3,3,3], ...]
# #             strides= [[1, 1, 1]] + [[2, 2, 2]] * (len(args.channels_list) - 1),                # e.g. [[1,1,1], [2,2,2], ...]
# #             upsample_kernel_size=[[2, 2, 2]] * (len(args.channels_list) - 1),  # same length as strides-1
# #             filters=args.filters if hasattr(args, "filters") else None,
# #             dropout=args.dropout_prob if args.dropout_prob > 0 else None,
# #             norm_name=("INSTANCE", {"affine": True}),   # or from args.norm if compatible
# #             act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
# #             deep_supervision=args.deep_supervision,
# #             deep_supr_num=args.deep_supr_num if hasattr(args, "deep_supr_num") else 1,
# #             res_block=args.res_block if hasattr(args, "res_block") else False,
# #             trans_bias=False,
# #         )
# #
# #     else:
# #         raise NotImplementedError(args.unet_arch + " is not implemented.")
