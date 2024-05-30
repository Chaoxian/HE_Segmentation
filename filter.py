import numpy as np
from scipy.ndimage import label, sum as ndi_sum
from PIL import Image
import torch
import cv2
import openslide
import slide
import glob
import colorsys

files = glob.glob("processed_image/test_*.png")
for file in files:

    Image.MAX_IMAGE_PIXELS = None
    # 读取mask图像，并将其转换为二值数组（假设mask部分为1，其他为0）
    mask_image = Image.open(file).convert('L')
    mask_array = np.array(mask_image)
    Image.fromarray((mask_array * 255).astype(np.uint8)).save(f"{file}_origin.png")
    binary_mask = (mask_array > 0).astype(np.uint8)

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # 定义最小区域大小
    min_region_size = 5000  # 根据需要调整

    # 标记连通区域
    labeled_mask, num_features = label(binary_mask)
    # import ipdb;ipdb.set_trace()
    # 计算每个区域的大小
    region_sizes = ndi_sum(binary_mask, labeled_mask, range(num_features + 1))
    # _,idx = torch.sort(torch.as_tensor(region_sizes),descending=True)
    # idx = idx[:20]
    # 计算平均值和标准差
    mean_size = np.mean(region_sizes)
    std_size = np.std(region_sizes)

    # 定义阈值，选择大于均值加若干倍标准差的区域
    threshold = mean_size + 0.5 * std_size
    # 创建一个新的mask，只保留大小达到一定值的区域
    filtered_mask = np.zeros_like(binary_mask)

    tmp = 0
    for i in range(1, num_features + 1):
        if region_sizes[i] >= max(threshold,min_region_size):
            binary_mask = np.zeros_like(filtered_mask,dtype=np.uint8)
            binary_mask[labeled_mask == i] = 1
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=10)
            filtered_mask[binary_mask == 1] = tmp+1
            tmp += 1

    colored_mask = np.zeros((*filtered_mask.shape, 3), dtype=np.uint8)
    num_colors = tmp 
    hue_values = np.linspace(0, 1, num_colors + 1)[:-1]
    colors = [colorsys.hsv_to_rgb(hue, 0.8, 0.8) for hue in hue_values]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in colors]
    for i in range(1, tmp+1):
        colored_mask[filtered_mask == i] = colors[i-1]

    file = file.split("/")[1].split("_")[1].split(".")[0]
    slide_img = openslide.OpenSlide("../HE_20240220/"+file+".ndpi")
    slide_image = np.array(slide.read_slide_at_mag(slide_img, 5).convert('RGB'))
    image_transparency = 0.6
    slide_img.close()

    slide_image = (slide_image * image_transparency + np.array(colored_mask) * (1-image_transparency)).astype(np.uint8)
    # 将结果保存为图像
    filtered_mask_image = Image.fromarray((colored_mask).astype(np.uint8))
    filtered_mask_image.save(f'{file}_filtered_mask.png')
    filtered_mask_image = Image.fromarray((slide_image).astype(np.uint8))
    filtered_mask_image.save(f'{file}_masked_image.png')
    print(tmp)
# 显示结果
# filtered_mask_image.show()
