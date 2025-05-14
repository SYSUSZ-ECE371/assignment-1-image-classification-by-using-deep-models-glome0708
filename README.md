## 实验一

- **数据划分脚本**  
  `EX1/split_dataset.py`

- **微调配置文件**  
  `EX1/configs/my_config/flower_resnet18_finetune.py`

- **预训练模型**  
  `EX1/checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth`

- **训练输出目录**  
  `EX1/work_dir/`

- **运行命令**  
  ```bash
  python EX1/tools/train.py EX1/configs/my_custom/flower_resnet18_finetune.py

## 实验二

- **程序输出**  
  `EX2/output.txt`

- **最佳模型文件**  
  `EX2/work_dir/best_model.pth`