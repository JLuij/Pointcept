_base_ = [
    # '../_base_/datasets/modelnet40.py',
    # '../_base_/schedulers/multi-step_sgd.py',
    # '../_base_/tests/classification.py',
    '../_base_/default_runtime.py'
]

## From authors @ https://github.com/Pointcept/PointTransformerV2/issues/9

train_gpu = [2, 3]

batch_size = 512
batch_size_val = 512
num_worker = 4
evaluate=True   # Evaluate after each epoch
metric = "allAcc"

# model = dict(
#     # type="PT-v2m2",
#     # in_channels=6,
#     # num_classes=40,
#     # channels=(48, 96, 192, 384, 512),
#     # patch_embed_depth=1,
#     # patch_embed_num_samples=8,
#     # patch_embed_group=4,
#     # enc_depths=(2, 2, 6, 2),
#     # dec_depths=(1, 1, 1, 1),
#     # down_stride=(4, 4, 4, 4),
#     # down_num_samples=(0.05, 0.1, 0.2, 0.4),  # Gird Size
#     # attn_groups=(12, 24, 48, 64),
#     # attn_num_samples=(16, 16, 16, 16),
#     # attn_qkv_bias=True,
#     # mlp_channels_expend_ratio=1.,
#     # drop_rate=0.,
#     # attn_drop_rate=0.,
#     # drop_path_rate=0.3,
    
#     type="PT-v3m1",
#     in_channels=6,
#     order=["z", "z-trans", "hilbert", "hilbert-trans"],
#     stride=(2, 2, 2, 2),
#     enc_depths=(2, 2, 2, 6, 2),
#     enc_channels=(48, 96, 192, 384, 512),
#     enc_num_head=(2, 4, 8, 16, 32),
#     enc_patch_size=(1024, 1024, 1024, 1024, 1024),
#     # dec_depths=(1, 1, 1, 1),
#     # dec_channels=(64, 64, 128, 256),
#     # dec_num_head=(4, 4, 8, 16),
#     # dec_patch_size=(1024, 1024, 1024, 1024),
#     mlp_ratio=4,
#     qkv_bias=True,
#     qk_scale=None,
#     attn_drop=0.0,
#     proj_drop=0.0,
#     drop_path=0.3,
#     shuffle_orders=True,
#     pre_norm=True,
#     enable_rpe=True,
#     enable_flash=False,
#     upcast_attention=False,
#     upcast_softmax=False,
#     cls_mode=True,
    
#     # pdnorm_bn=False,
#     # pdnorm_ln=False,
#     # pdnorm_decouple=True,
#     # pdnorm_adaptive=False,
#     # pdnorm_affine=True,
#     # pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
# )

model = dict(
    type="DefaultClassifier",
    num_classes=10,
    backbone_embed_dim=8,
    backbone=dict(
        type="PT-v3m1",
        in_channels=128,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride = (2,),
        enc_depths = (2, 2),
        enc_channels = (4, 8),
        enc_num_head = (2, 4),
        enc_patch_size = (16, 16),
        # dec_depths=(1, 1, 1, 1),
        # dec_channels=(64, 64, 128, 256),
        # dec_num_head=(4, 4, 8, 16),
        # dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=True,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

epochs = 100
start_epoch = 0
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type='MultiStepLR', milestones=[epochs * 0.6, epochs * 0.8], gamma=0.1)
# scheduler = dict(type='MultiStepLR', milestones=[epochs * 0.6, epochs * 0.8], steps_per_epoch=1, gamma=0.1)


# dataset settings
dataset_type = "PointINRDataSet"
data_root = "data/pointINR_centered"
shared_decoder_path = True
cache_data = True
names = ["0","1","2","3","4","5","6","7","8","9"]

data = dict(
    num_classes=10,
    ignore_label=-1,  # dummy ignore
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=names,
        transform=[
            dict(type="NormalizeCoord"),
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis='x', p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis='y', p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[[0.2, 0.2]]
            #                                [[0.2, 0.2]]
            #                                [[0.2, 0.2]]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),

            # dict(type="Voxelize", voxel_size=0.01, hash_type='fnv', mode='train'),
            # dict(type="SphereCrop", point_max=10000, mode='random'),
            # dict(type="CenterShift", apply_z=True),
            dict(type="ShufflePoint"),
            dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # dict(type="Copy", keys_dict={"offset": -1}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "category", "grid_size"),
                feat_keys=["feats"],
                # feat_keys=["coord", "feats"],
            ),
        ],
        loop=2,
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="Copy", keys_dict={"grid_size": 0.01}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "category", "grid_size"),
                feat_keys=["feats"],
                # feat_keys=["coord", "feats"],
            ),
        ],
        loop=1,
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClassificationEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# criteria = [
#     dict(type="CrossEntropyLoss",
#          loss_weight=1.0,
#          ignore_index=data["ignore_label"])
# ]