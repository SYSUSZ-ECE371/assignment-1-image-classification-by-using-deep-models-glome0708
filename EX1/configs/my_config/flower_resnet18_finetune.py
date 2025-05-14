_base_ = ['mmcls::resnet/resnet18_8xb32_in1k.py']

#Modify the model configuration
model = dict(
    head=dict(
        num_classes=5,
    )
)

#Modify the dataset configuration
dataset_type = 'ImageNet'
data_root = 'EX1/imagenet_format'
classes_file = 'EX1/imagenet_format/classes.txt'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        classes=classes_file,
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        classes=classes_file,
    )
)

val_evaluator = dict(
    topk=(1,)
)

#Modify learning rate strategy
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001
    )
)

train_cfg = dict(by_epoch=True, max_epochs=10)

#Configuring pre-trained models
load_from = 'EX1/checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'

work_dir = 'EX1/work_dir'