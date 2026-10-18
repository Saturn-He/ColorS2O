import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2
import torchvision.transforms as transforms

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy


from util.datasets import ImageDirDataset, PairedImageDirDataset

def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if len(batch) == 4:
            sar_img, opt_img, hint_color, hint_mask = batch
        elif len(batch) == 2:
            sar_img, opt_img = batch
        else:
            raise ValueError("Unexpected batch format for training.")

        # normalize image to [-1, 1]
        sar_img = sar_img.to(device, non_blocking=True).to(torch.float32).div_(255)
        sar_img = sar_img * 2.0 - 1.0
        opt_img = opt_img.to(device, non_blocking=True)
        opt_img = opt_img.to(torch.float32).div_(255)
        opt_img = opt_img * 2.0 - 1.0
        hint_color = hint_color.to(device, non_blocking=True).to(torch.float32).div_(255)
        hint_color = hint_color * 2.0 - 1.0
        hint_mask = hint_mask.to(device, non_blocking=True).to(torch.float32)
        hint_input = torch.cat([hint_color, hint_mask], dim=1)
        labels = torch.zeros(opt_img.size(0), device=device, dtype=torch.long)

        # epoch schedule
        p = epoch / max(args.epochs - 1, 1)

        if args.epoch_schedule == "linear":
            epoch_lambda = p
        elif args.epoch_schedule == "cosine":
            epoch_lambda = 0.5 * (1 - math.cos(math.pi * p))
        else:
            epoch_lambda = 1.0

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(opt_img, sar_img, labels, hint_input=hint_input, hint_mask=hint_mask, epoch_lambda=epoch_lambda)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.sync_cuda_every_step:
            torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()

    transform_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.PILToTensor()
    ])
    if args.use_hint_infer:
        if not args.opt_test_path:
            raise ValueError("opt_test_path is required when use_hint_infer is enabled.")
        sar_dataset = PairedImageDirDataset(
            args.sar_test_path,
            args.opt_test_path,
            hint_root=args.hint_test_path or None,
            transform=transform_eval,
            return_names=True,
        )
    else:
        sar_dataset = ImageDirDataset(args.sar_test_path, transform=transform_eval, mode="L")
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(
            sar_dataset, num_replicas=world_size, rank=local_rank, shuffle=False
        )
    else:
        sampler = torch.utils.data.SequentialSampler(sar_dataset)
    data_loader = torch.utils.data.DataLoader(
        sar_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    num_images = min(args.num_images, len(sar_dataset))
    num_steps = (num_images // (batch_size * world_size)) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    os.makedirs(save_folder, exist_ok=True)
    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    img_count = 0
    for i, batch in enumerate(data_loader):
        if img_count >= num_images:
            break
        print("Generation step {}/{}".format(i, num_steps))

        if args.use_hint_infer:
            sar_img, opt_img, hint_color, hint_mask, sar_names = batch
        else:
            sar_img, sar_names = batch

        sar_img = sar_img.to(torch.device(args.device))
        sar_img = sar_img.to(torch.float32).div_(255)
        sar_img = sar_img * 2.0 - 1.0
        labels_gen = torch.zeros(sar_img.size(0), device=sar_img.device, dtype=torch.long)
        hint_input = None
        if args.use_hint_infer:
            hint_color = hint_color.to(args.device, non_blocking=True).to(torch.float32).div_(255)
            hint_color = hint_color * 2.0 - 1.0
            hint_mask = hint_mask.to(args.device, non_blocking=True).to(torch.float32)
            hint_input = torch.cat([hint_color, hint_mask], dim=1)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(sar_img, labels_gen, hint_input=hint_input)

        if args.distributed:
            torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = img_count + b_id
            if img_id >= num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, sar_names[b_id]), gen_img)
        img_count += sampled_images.size(0)

    if args.distributed:
        torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        if not args.keep_outputs:
            shutil.rmtree(save_folder)

    if args.distributed:
        torch.distributed.barrier()
