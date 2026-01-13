import lpips
import torch
import torch.nn as nn
from kornia.color import rgb_to_lab
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            out_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.enabled_losses = set(args.enabled_losses)
        self.lambda_v = args.lambda_v
        self.lambda_ab = args.lambda_ab
        self.lambda_perc = args.lambda_perc
        self.lambda_sam = args.lambda_sam
        self.loss_eps = 1e-6
        self.lpips_model = None
        if "perc" in self.enabled_losses:
            self.lpips_model = lpips.LPIPS(net="vgg")
            self.lpips_model.eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def _apply_sample_weights(self, loss_per_sample, weight, name):
        weight_tensor = torch.as_tensor(weight, device=loss_per_sample.device, dtype=loss_per_sample.dtype)
        if weight_tensor.ndim == 0:
            weight_tensor = weight_tensor.expand_as(loss_per_sample)
        elif weight_tensor.ndim == 1 and weight_tensor.shape[0] == loss_per_sample.shape[0]:
            weight_tensor = weight_tensor
        else:
            raise ValueError(f"{name} must be a scalar or a 1D tensor with batch size.")
        return (loss_per_sample * weight_tensor).mean()
        
    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, opt_img, sar_img, labels=None):
        if labels is None:
            labels = torch.zeros(opt_img.size(0), device=opt_img.device, dtype=torch.long)
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(opt_img.size(0), device=opt_img.device).view(-1, *([1] * (opt_img.ndim - 1)))
        e = torch.randn_like(opt_img) * self.noise_scale

        z = t * opt_img + (1 - t) * e
        v = (opt_img - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped, sar_img)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # v loss (l2)
        v_loss = (v - v_pred) ** 2
        v_loss = v_loss.mean(dim=(1, 2, 3))
        loss = self._apply_sample_weights(v_loss, self.lambda_v, "lambda_v")

        if "ab" in self.enabled_losses:
            x_lab = rgb_to_lab(x_pred.clamp(0.0, 1.0))
            opt_lab = rgb_to_lab(opt_img.clamp(0.0, 1.0))
            ab_loss = torch.abs(x_lab[:, 1:, ...] - opt_lab[:, 1:, ...]).mean(dim=(1, 2, 3))
            loss = loss + self._apply_sample_weights(ab_loss, self.lambda_ab, "lambda_ab")

        if "perc" in self.enabled_losses and self.lpips_model is not None:
            x_norm = x_pred.clamp(-1.0, 1.0)
            opt_norm = opt_img.clamp(-1.0, 1.0)
            perc_loss = self.lpips_model(x_norm, opt_norm).view(opt_img.size(0), -1).mean(dim=1)
            loss = loss + self._apply_sample_weights(perc_loss, self.lambda_perc, "lambda_perc")

        if "sam" in self.enabled_losses:
            dot = (x_pred * opt_img).sum(dim=1)
            denom = torch.norm(x_pred, dim=1) * torch.norm(opt_img, dim=1)
            denom = denom.clamp_min(self.loss_eps)
            cos_angle = dot / denom
            cos_angle = torch.clamp(cos_angle, -1.0 + self.loss_eps, 1.0 - self.loss_eps)
            sam_loss = torch.acos(cos_angle).mean(dim=(1, 2))
            loss = loss + self._apply_sample_weights(sam_loss, self.lambda_sam, "lambda_sam")

        return loss

    @torch.no_grad()
    def generate(self, sar_img, labels=None):
        if labels is None:
            labels = torch.zeros(sar_img.size(0), device=sar_img.device, dtype=torch.long)
        device = sar_img.device
        bsz = sar_img.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels, sar_img)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, sar_img)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, sar_img):
        # conditional
        x_cond = self.net(z, t.flatten(), labels, sar_img)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes), sar_img)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, sar_img):
        v_pred = self._forward_sample(z, t, labels, sar_img)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, sar_img):
        v_pred_t = self._forward_sample(z, t, labels, sar_img)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, sar_img)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
