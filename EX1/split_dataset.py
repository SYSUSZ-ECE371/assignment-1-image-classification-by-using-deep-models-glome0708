import os
import random
import shutil

random.seed(1234)

# 原始数据路径
source_dir = 'EX1/flower_dataset'

# 目标根路径
target_root = 'EX1/imagenet_format'
train_dir = os.path.join(target_root, 'train')
val_dir = os.path.join(target_root, 'val')

# 创建目标目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取并排序类别
classes = sorted(os.listdir(source_dir))

# 写入 classes.txt
with open(os.path.join(target_root, 'classes.txt'), 'w') as f:
    for cls in classes:
        f.write(cls + '\n')

# 打开 annotation 文件
train_txt = open(os.path.join(target_root, 'train.txt'), 'w')
val_txt = open(os.path.join(target_root, 'val.txt'), 'w')

for idx, cls in enumerate(classes):
    src_path = os.path.join(source_dir, cls)
    train_cls_dir = os.path.join(train_dir, cls)
    val_cls_dir = os.path.join(val_dir, cls)
    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(val_cls_dir, exist_ok=True)

    images = os.listdir(src_path)
    random.shuffle(images)

    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    val_images = images[split_index:]

    # 重命名计数器
    train_count = 1
    val_count = 1

    for img in train_images:
        ext = os.path.splitext(img)[1]  # 保留原始扩展名
        new_name = f"NAME{train_count:03d}{ext}"
        train_count += 1

        src = os.path.join(src_path, img)
        dst = os.path.join(train_cls_dir, new_name)
        shutil.copy2(src, dst)

        train_txt.write(f"{cls}/{new_name} {idx}\n")

    for img in val_images:
        ext = os.path.splitext(img)[1]
        new_name = f"NAME{val_count:03d}{ext}"
        val_count += 1

        src = os.path.join(src_path, img)
        dst = os.path.join(val_cls_dir, new_name)
        shutil.copy2(src, dst)

        val_txt.write(f"{cls}/{new_name} {idx}\n")

train_txt.close()
val_txt.close()

print("ImageNet已生成")