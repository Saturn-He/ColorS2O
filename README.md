# Just image Transformer (JiT) for SAR-to-Optical Image Translation

## Train：

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 main_jit.py

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master-port=29506 main_jit.py \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT/round1" \
  --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA" \
  --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB" \
  --img_size 512

### Train with different lr:

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29505 main_jit.py \
  --blr 1.6e-3 \
  --output_dir /NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/lr5em5

### Train on GF3:

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master-port=29501 main_jit.py --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/GF3/caJiT/round1" --sar_train_path="/NAS_data/yjy/GF3_High_Res/trainA" --opt_train_path="/NAS_data/yjy/GF3_High_Res/trainB" --img_size=256 --resume "/NAS_data/hjf/JiTcolor/checkpoints/GF3/caJiT/round1"

### Train on SAR2Opt:

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master-port=29506 main_jit.py   --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT/round1"   --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA"   --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB"   --img_size 512 --resume "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT/round1"

### EP1：v_loss + L_ab + L_perc + L_sam

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master-port=29501 main_jit.py   --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_loss_only/round2_progressive/ab_perc_sam"   --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA"   --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB"  --img_size 512  --enabled_losses ab perc sam

### EP2：v_loss + L_ab

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master-port=29502 main_jit.py   --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_loss_only/round2_progressive/ab"   --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA"   --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB"  --img_size 512  --enabled_losses ab

### EP3：v_loss + L_perc

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master-port=29503 main_jit.py   --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_loss_only/round2_progressive/perc"   --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA"   --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB"  --img_size 512  --enabled_losses perc

### EP4：v_loss + L_sam

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master-port=29504 main_jit.py   --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_loss_only/round2_progressive/sam"   --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA"   --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB"  --img_size 512  --enabled_losses sam


### Train on SEN-SCENE:

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master-port=29504 main_jit.py --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/scene/caJiT/round1" --sar_train_path="/data/hjf/Dataset/SEN12_Scene/trainA" --opt_train_path="/data/hjf/Dataset/SEN12_Scene/trainB" --img_size=256 --resume "/NAS_data/hjf/JiTcolor/checkpoints/scene/caJiT/round1"


## Train with model JiT-L/16

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master-port=29508 main_jit.py \
  --model JiT-L/16 \
  --img_size 512 \
  --batch_size 4 \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT/JiT-L16" \
  --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA" \
  --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB" \
  --resume "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT/JiT-L16"


## Train with model JiT-H/16

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master-port=29507 main_jit.py \
  --model JiT-H/16 \
  --img_size 512 \
  --batch_size 2 \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT/JiT-H16" \
  --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA" \
  --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB" \
  --resume "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT/JiT-H16"


## Inference：

### Inference on SAR2Opt：

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29503 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt --sar_test_path /NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/SAR2Opt/round1 --img_size 512 --gen_bsz 8 --keep_outputs

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29507 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT/round3 --sar_test_path /NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/SAR2Opt/caJiT/round3 --img_size 512 --gen_bsz 8 --keep_outputs

### EP1：v_loss + L_ab + L_perc + L_sam

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master-port=29501 main_jit.py   --resume "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_loss_only/round2_progressive/ab_perc_sam"   --sar_test_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA"   --img_size 512  --enabled_losses ab perc sam  --evaluate_gen  --gen_bsz 8 --keep_outputs  --output_dir "/NAS_data/hjf/JiTcolor/outputs/SAR2Opt/caJiT_loss_only/round2_progressive/ab_perc_sam"

### EP2：v_loss + L_ab

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master-port=29502 main_jit.py   --resume "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_loss_only/round2_progressive/ab"   --sar_test_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA"   --img_size 512  --enabled_losses ab  --evaluate_gen  --gen_bsz 8 --keep_outputs  --output_dir "/NAS_data/hjf/JiTcolor/outputs/SAR2Opt/caJiT_loss_only/round2_progressive/ab"

### EP3：v_loss + L_perc

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master-port=29503 main_jit.py   --resume "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_loss_only/round2_progressive/perc"   --sar_test_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA"   --img_size 512  --enabled_losses perc  --evaluate_gen  --gen_bsz 8 --keep_outputs  --output_dir "/NAS_data/hjf/JiTcolor/outputs/SAR2Opt/caJiT_loss_only/round2_progressive/perc"

### EP4：v_loss + L_sam

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master-port=29504 main_jit.py   --resume "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_loss_only/round2_progressive/sam"   --sar_test_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA"   --img_size 512  --enabled_losses sam  --evaluate_gen  --gen_bsz 8 --keep_outputs  --output_dir "/NAS_data/hjf/JiTcolor/outputs/SAR2Opt/caJiT_loss_only/round2_progressive/sam"


### Inference on GF3：

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29504 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/GF3 --sar_test_path /NAS_data/yjy/GF3_High_Res/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/GF3/round1 --img_size 256 --gen_bsz 8 --keep_outputs

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29508 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/GF3/caJiT/round3 --sar_test_path /NAS_data/yjy/GF3_High_Res/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/GF3/caJiT/round3 --img_size 256 --gen_bsz 8 --keep_outputs

### Inference on SEN-SCENE：

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29505 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/scene --sar_test_path /data/hjf/Dataset/SEN12_Scene/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/scene/round1 --img_size 256 --gen_bsz 8 --keep_outputs

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29508 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/scene/caJiT/round3 --sar_test_path /data/hjf/Dataset/SEN12_Scene/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/scene/caJiT/round3 --img_size 256 --gen_bsz 8 --keep_outputs
