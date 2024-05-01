
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt

import numpy as np
import SimpleITK as sitk
import os
import numpy as np
from glob import glob
import time
import shutil
from PIL import Image


def extract_amp_spectrum(trg_img):
    fft_trg_np = np.fft.fft2(trg_img, axes=(-2, -1))
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target


def amp_spectrum_swap(amp_local, amp_target, L=0.1, ratio=0):
    a_local = np.fft.fftshift(amp_local, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_target, axes=(-2, -1))

    _, h, w = a_local.shape


    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_local[:, h1:h2, w1:w2] = a_local[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)

    a_local = np.fft.ifftshift(a_local, axes=(-2, -1))
    return a_local


def freq_space_interpolation(local_img, amp_target, L=0, ratio=0):
    local_img_np = local_img

    # get fft of local sample
    fft_local_np = np.fft.fft2(local_img_np, axes=(-2, -1))

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap(amp_local, amp_target, L=L, ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp(1j * pha_local)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = np.real(local_in_trg)

    return local_in_trg

def calculate_average_spectrum(folder_path):
    file_list = os.listdir(folder_path)
    total_spectrum = None
    count = 0

    for file_name in file_list:
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            img_path = os.path.join(folder_path, file_name)
            img = Image.open(img_path)
            img_np = np.asarray(img, np.float32)
            img_np = img_np.transpose((2, 0, 1))  # 转换图像格式

            # 计算幅度谱
            fft_img_np = np.fft.fft2(img_np, axes=(-2, -1))
            amp_img = np.abs(fft_img_np)

            # 累加幅度谱
            if total_spectrum is None:
                total_spectrum = amp_img
            else:
                total_spectrum += amp_img

            count += 1

    # 计算平均幅度谱
    average_spectrum = total_spectrum / count if count > 0 else None
    return average_spectrum


# def extract_all_amp_spectra(folder_path):
#     file_list = os.listdir(folder_path)
#     amp_spectra = []
#
#     for file_name in file_list:
#         if file_name.endswith('.jpg') or file_name.endswith('.png'):
#             img_path = os.path.join(folder_path, file_name)
#             img = Image.open(img_path)
#             img_np = np.asarray(img, np.float32)
#             img_np = img_np.transpose((2, 0, 1))  # 转换图像格式
#
#             # 计算幅度谱并添加到列表中
#             fft_img_np = np.fft.fft2(img_np, axes=(-2, -1))
#             amp_img = np.abs(fft_img_np)
#             amp_spectra.append(amp_img)
#
#     return amp_spectra

def draw_image(image):
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    plt.xticks([])
    plt.yticks([])

    return 0


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def freq_transform_func(x, domain):
    target_path = (f"H:\workspace\data\Segmentation\Fundus\\train\Domain{domain}")
    x = np.asarray(x, np.float32).transpose(2,0,1)
    def random_spec():
        import random
        files = os.listdir(target_path)
        file = files[random.randint(0, len(files)-1)]
        img = Image.open(os.path.join(target_path,file))
        img_np = np.asarray(img, np.float32)
        img_np = img_np.transpose((2, 0, 1))  # 转换图像格式

        # 计算幅度谱
        fft_img_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp_img = np.abs(fft_img_np)
        return  amp_img
    amp_target = random_spec()
    transformed_x = freq_space_interpolation(x, amp_target, L=0.003, ratio=0)
    transformed_x = transformed_x.transpose(1,2,0).astype('uint8')
    print(transformed_x.dtype)
    transformed_x = Image.fromarray(transformed_x)
    return transformed_x


if __name__ == '__main__':
    im_local_path = ("H:\workspace\data\Segmentation\Fundus\Domain3_all")
    amp_target = calculate_average_spectrum("H:\workspace\data\Segmentation\Fundus\Domain1_all")
    output_folder = "H:\workspace\data\Segmentation\Fundus\Domain3\\freq_0.4_TEST"

    # im_local_path = r"I:\phd\Data\UHU\images_h_test\benign"
    # amp_target = calculate_average_spectrum(r"I:\phd\Data\UHU\test_all")
    # output_folder = r"I:\phd\Data\UHU\images_h_f0_test\benign"

    # im_local_path = r"G:\workspace\data\PcrUK\PcrUK_train_h\N"
    # amp_target = calculate_average_spectrum(r"G:\workspace\data\PcrUK\PcrUK_train\N")
    # output_folder = r"G:\workspace\data\PcrUK\PcrUK_train_f\N"

    os.makedirs(output_folder, exist_ok=True)

    for idx, im_local_name in enumerate(os.listdir(im_local_path)):
        im_local = Image.open(os.path.join(im_local_path,im_local_name))
        im_local = np.asarray(im_local, np.float32)
        im_local = im_local.transpose((2, 0, 1))
        # 进行频率空间插值
        interpolated_img = freq_space_interpolation(im_local, amp_target, L=0.003, ratio=0)
        interpolated_img = interpolated_img.transpose((1, 2, 0))
        # 保存插值结果
        output_filename = f"0.4_{im_local_name}"
        output_path = os.path.join(os.path.join(output_folder, output_filename))
        # interpolated_img = Image.fromarray((interpolated_img * 255).astype(np.uint8))
        interpolated_img = Image.fromarray((interpolated_img).astype(np.uint8))
        interpolated_img.save(output_path)

        # # continuous frequency space interpolation
        # for idx, i in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        #     plt.subplot(1, 8, idx + 4)
        #     local_in_trg = freq_space_interpolation(im_local, amp_target, L=L, ratio=1 - i)
        #     local_in_trg = local_in_trg.transpose((1, 2, 0))
        #     draw_image((np.clip(local_in_trg / 255, 0, 1)))
        #     plt.xlabel("Interpolation Rate: {}".format(i), fontsize=12)
        # plt.show()




