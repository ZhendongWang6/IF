from utils import util_calculate_psnr_ssim as util
from PIL import Image
import numpy as np
import glob
import os
from collections import OrderedDict


def make_resizer(library, quantize_after, filter, output_size):
    if library == "PIL" and quantize_after:
        name_to_filter = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX,
        }

        def func(x):
            x = Image.fromarray(x)
            x = x.resize(
                output_size[::-1], resample=name_to_filter[filter]
            )  # resize((w,h))
            x = np.asarray(x).clip(0, 255).astype(np.uint8)
            return x

    elif library == "PIL" and not quantize_after:
        name_to_filter = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX,
        }
        s1, s2 = output_size

        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode="F")
            img = img.resize(
                output_size[::-1], resample=name_to_filter[filter]
            )  # resize((w,h))
            return np.asarray(img).clip(0, 255).reshape(s1, s2, 1)

        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x

    elif library == "PyTorch":
        import warnings

        # ignore the numpy warnings
        warnings.filterwarnings("ignore")

        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x

    elif library == "TensorFlow":
        import warnings

        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        import tensorflow as tf

        def func(x):
            x = tf.constant(x)[tf.newaxis, ...]
            x = tf.image.resize(x, output_size, method=filter)
            x = x[0, ...].numpy().clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x

    elif library == "OpenCV":
        import cv2

        name_to_filter = {
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
        }

        def func(x):
            x = cv2.resize(x, output_size, interpolation=name_to_filter[filter])
            x = x.clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x

    else:
        raise NotImplementedError("library [%s] is not include" % library)
    return func


def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = torch.from_numpy(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = "cubic"

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing
    )
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing
    )
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = (
                img_aug[idx : idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])
            )

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx : idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()


border = 4
HR_path = sorted(glob.glob("testsets/Set5/HR/*"))
LR_Path = sorted(glob.glob("testsets/Set5/LR_bicubic/X4/*"))
test_results = OrderedDict()
test_results["psnr"] = []
test_results["ssim"] = []
test_results["psnr_y"] = []
test_results["ssim_y"] = []

for LR, HR in zip(LR_Path, HR_path):
    img_gt = np.array(Image.open(HR))
    img_lr = np.array(Image.open(LR))
    fn = os.path.splitext(os.path.basename(HR))[0]
    resizer = make_resizer("PIL", False, "bicubic", img_gt.shape[:2])
    output = resizer(img_lr)
    # output = np.array(Image.open("results/deepif_classical_sr_x4/baby_DeepIF.png"))

    
    psnr = util.calculate_psnr(output, img_gt, crop_border=border)
    ssim = util.calculate_ssim(output, img_gt, crop_border=border)
    psnr_y = util.calculate_psnr(
        output, img_gt, crop_border=border, test_y_channel=True
    )
    ssim_y = util.calculate_ssim(
        output, img_gt, crop_border=border, test_y_channel=True
    )

    print("Testing {} - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}.".format(
            fn, psnr, ssim, psnr_y, ssim_y))

test_results["psnr"].append(psnr)
test_results["ssim"].append(ssim)
test_results["psnr_y"].append(psnr_y)
test_results["ssim_y"].append(ssim_y)
ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])

print("\n\n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}".format(ave_psnr, ave_ssim))
print("-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}".format(ave_psnr_y, ave_ssim_y))
