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

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master-port=29506 main_jit.py --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/GF3" --sar_train_path="/NAS_data/yjy/GF3_High_Res/trainA" --opt_train_path="/NAS_data/yjy/GF3_High_Res/trainB" --sar_test_path="/NAS_data/yjy/GF3_High_Res/testA" --opt_test_path="/NAS_data/yjy/GF3_High_Res/testB" --img_size=256

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29502 main_jit.py \
  --sar_train_path "/NAS_data/yjy/GF3_High_Res/trainA" \
  --opt_train_path "/NAS_data/yjy/GF3_High_Res/trainB" \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/GF3/caJiT_CP/round2_inference_with_cp" \
  --img_size 256 \
  --hint_dropout_prob 0.5 \
  --hint_max_ratio 0.05 \
  --hint_color_thresh 0.1 \
  --hint_num_regions 4 \
  --hint_loss_weight 2.0

### Train on SAR2Opt:

CUDA_VISIBLE_DEVICES=1,4 torchrun --nproc_per_node=2 --master-port=29509 main_jit.py \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_CP/round3" \
  --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA" \
  --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB" \
  --img_size 512 \
  --hint_dropout_prob 0.5 \
  --hint_max_ratio 0.05 \
  --hint_color_thresh 0.1 \
  --hint_num_regions 4 \
  --hint_loss_weight 2.0

CUDA_VISIBLE_DEVICES=2,3,5,7 torchrun --nproc_per_node=4 --master-port=29509 main_jit.py \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_CP/round3" \
  --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA" \
  --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB" \
  --img_size 512 \
  --hint_dropout_prob 0.5 \
  --hint_max_ratio 0.05 \
  --hint_color_thresh 0.1 \
  --hint_num_regions 4 \
  --hint_loss_weight 2.0 \
  --batch_size 2

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master-port=29505 main_jit.py \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_CP/round4/noLoss_noHintsDropout_stripe_concat" \
  --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA" \
  --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB" \
  --img_size 512 \
  --hint_dropout_prob 0 \
  --hint_loss_weight 0 \
  --hint_sampling_mode stripe \
  --hint_on_gpu

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master-port=29504 main_jit.py \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_CP/round4/noLoss_noHintsDropout_dot_concat" \
  --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA" \
  --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB" \
  --img_size 512 \
  --hint_dropout_prob 0 \
  --hint_loss_weight 0 \
  --hint_sampling_mode dot \
  --hint_on_gpu


### Train on scene:

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29505 main_jit.py \
  --sar_train_path "/data/hjf/Dataset/SEN12_Scene/trainA" \
  --opt_train_path "/data/hjf/Dataset/SEN12_Scene/trainB" \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/scene/caJiT_CP/round3_inference_with_cp" \
  --img_size 256 \
  --hint_dropout_prob 0.5 \
  --hint_max_ratio 0.05 \
  --hint_color_thresh 0.1 \
  --hint_num_regions 4 \
  --hint_loss_weight 2.0

## Train with model JiT-L/16

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master-port=29508 main_jit.py \
  --model JiT-L/16 \
  --img_size 512 \
  --batch_size 4 \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT/JiT-L16" \
  --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA" \
  --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB"


## Train with model JiT-H/16

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master-port=29507 main_jit.py \
  --model JiT-H/16 \
  --img_size 512 \
  --batch_size 2 \
  --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT/JiT-H16" \
  --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA" \
  --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB"


## Inference：

### Inference on SAR2Opt：

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29503 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt --sar_test_path /NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/SAR2Opt/round1 --img_size 512 --gen_bsz 8 --keep_outputs

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master-port=29508 main_jit.py \
--model JiT-L/16 \
--img_size 512 \
--batch_size 4 \
--output_dir "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/controlJiT/JiT-L16" \
--sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainA" \
--opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/trainB"

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29503 main_jit.py \
  --evaluate_gen \
  --resume "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_CP/round1" \
  --sar_test_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA" \
  --opt_test_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testB" \
  --output_dir "/NAS_data/hjf/JiTcolor/outputs/caJiT/round2_inference_with_cp" \
  --img_size 512 \
  --gen_bsz 8 \
  --keep_outputs \
  --use_hint_infer \
  --hint_dropout_prob 0.5 \
  --hint_max_ratio 0.05 \
  --hint_color_thresh 0.1 \
  --hint_num_regions 4

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master-port=29509 main_jit.py \
  --evaluate_gen \
  --gen_bsz 8 \
  --keep_outputs \
  --resume "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_CP/round3" \
  --output_dir "/NAS_data/hjf/JiTcolor/outputs/SAR2Opt/caJiT_CP/round3" \
  --sar_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA" \
  --opt_train_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testB" \
  --img_size 512 \
  --use_hint_infer \
  --hint_dropout_prob 0.5 \
  --hint_max_ratio 0.05 \
  --hint_color_thresh 0.1 \
  --hint_num_regions 4 

### Inference on GF3：

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29504 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/GF3 --sar_test_path /NAS_data/yjy/GF3_High_Res/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/GF3/round1 --img_size 256 --gen_bsz 8 --keep_outputs

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29503 main_jit.py \
  --evaluate_gen \
  --resume "/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_CP/round1" \
  --sar_test_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA" \
  --opt_test_path "/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testB" \
  --output_dir "/NAS_data/hjf/JiTcolor/outputs/caJiT/round2_inference_with_cp" \
  --img_size 512 \
  --gen_bsz 8 \
  --keep_outputs \
  --use_hint_infer \
  --hint_dropout_prob 0.5 \
  --hint_max_ratio 0.05 \
  --hint_color_thresh 0.1 \
  --hint_num_regions 4

### Inference on SEN-SCENE：

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29505 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/scene --sar_test_path /data/hjf/Dataset/SEN12_Scene/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/scene/round1 --img_size 256 --gen_bsz 8 --keep_outputs
