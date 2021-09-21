import PIL.Image as Image
import os
from torchvision import transforms as transforms
import torchvision.transforms.functional as TF

"""定义数据增强方法：裁剪、翻转"""
'''----------------------------------------------------------------------'''


def read_PIL(image_path):
    image = Image.open(image_path)
    return image


# 五角裁剪
def five_crop_2048_1500(image):
    FiveCrop = transforms.FiveCrop(size=(2048, 1500))
    five_crop_images = list(FiveCrop(image))
    return five_crop_images


def five_crop_1024_750(image):
    FiveCrop = transforms.FiveCrop(size=(1024, 750))
    five_crop_images = list(FiveCrop(image))
    return five_crop_images


def five_crop_512_375(image):
    FiveCrop = transforms.FiveCrop(size=(512, 375))
    five_crop_images = list(FiveCrop(image))
    return five_crop_images


def five_crop_256_224(image):
    FiveCrop = transforms.FiveCrop(size=(256, 224))
    five_crop_images = list(FiveCrop(image))
    return five_crop_images


# 中心裁剪
def center_crop_1224_1024(image):
    CenterCrop = transforms.CenterCrop(size=(1224, 1024))
    cropped_image = CenterCrop(image)
    return cropped_image


def five_crop_612_512(image):
    FiveCrop = transforms.FiveCrop(size=(612, 512))
    five_crop_images = list(FiveCrop(image))
    return five_crop_images


def five_crop_306_256(image):
    FiveCrop = transforms.FiveCrop(size=(306, 256))
    five_crop_images = list(FiveCrop(image))
    return five_crop_images


# 垂直翻转
def vertical_flip(image):
    VF = transforms.RandomVerticalFlip()
    vf_image = VF(image)
    return vf_image


# 水平翻转
def horizontal_flip(image):
    HF = transforms.RandomHorizontalFlip()
    hf_image = HF(image)
    return hf_image


# 随机角度旋转
def random_rotation(image):
    RR = transforms.RandomRotation(degrees=(10, 80))
    rr_image = RR(image)
    return rr_image


'''----------------------------------------------------------------------'''

'''处理图片并保存结果'''

'''----------------------------------------------------------------------'''


def centercrop(outDir):
    os.makedirs(outDir, exist_ok=True)
    file_dir = os.listdir(outDir)
    for file in file_dir:
        file_name = os.path.join(outDir, file)
        im = read_PIL(file_name)
        if file.split('.')[1] == 'jpg':
            center_cropped_image = center_crop_1224_1024(im)
            center_cropped_image.save(os.path.join(outDir, file))


def fivecrop(outDir):
    # 裁剪
    os.makedirs(outDir, exist_ok=True)
    file_dir = os.listdir(outDir)
    for i in range(4):
        f = [five_crop_2048_1500, five_crop_1024_750, five_crop_512_375, five_crop_256_224]
        g = [five_crop_612_512, five_crop_306_256, five_crop_256_224, five_crop_256_224]
        file_dir = os.listdir(outDir)
        for file in file_dir:
            augmentation = f if file.split('.')[1] == 'bmp' else g
            file_name = os.path.join(outDir, file)
            im = read_PIL(file_name)
            images = augmentation[i](im)
            for j in range(5):
                images[j].save(
                    os.path.join(outDir, file.split('.')[0] + "_" + str(j + 1) + "." + str(file.split('.')[1])))
            os.remove(file_name)


'''----------------------------------------------------------------------'''

'''处理18/21张的数据集 扩容2500倍'''


def augmentation_2500(outDir):
    def flip(outDir):
        os.makedirs(outDir, exist_ok=True)
        file_dir = os.listdir(outDir)
        for file in file_dir:
            file_name = os.path.join(outDir, file)
            im = read_PIL(file_name)

            hf_image = horizontal_flip(im)  # 水平翻转
            hf_image.save(os.path.join(outDir, file.split('.')[0] + '_hf_image.jpg'))

            vf_image = vertical_flip(im)  # 垂直翻转
            vf_image.save(os.path.join(outDir, file.split('.')[0] + '_vf_image.jpg'))

            rr_image = random_rotation(im)  # 随机翻转
            rr_image.save(os.path.join(outDir, file.split('.')[0] + '_rr_image.jpg'))

    centercrop(outDir)
    fivecrop(outDir)
    flip(outDir)


'''处理30/40张的数据集 扩容1875倍'''


def augmentation_1875(outDir):
    def flip(outDir):
        os.makedirs(outDir, exist_ok=True)
        file_dir = os.listdir(outDir)
        for file in file_dir:
            file_name = os.path.join(outDir, file)
            im = read_PIL(file_name)

            hf_image = horizontal_flip(im)  # 水平翻转
            hf_image.save(os.path.join(outDir, file.split('.')[0] + '_hf_image.jpg'))

            vf_image = vertical_flip(im)  # 垂直翻转
            vf_image.save(os.path.join(outDir, file.split('.')[0] + '_vf_image.jpg'))

    centercrop(outDir)
    fivecrop(outDir)
    flip(outDir)


'''处理46张的数据集 扩容1250倍'''


def augmentation_1250(outDir):
    def flip(outDir):
        os.makedirs(outDir, exist_ok=True)
        file_dir = os.listdir(outDir)
        for file in file_dir:
            file_name = os.path.join(outDir, file)
            im = read_PIL(file_name)

            vf_image = vertical_flip(im)  # 垂直翻转
            vf_image.save(os.path.join(outDir, file.split('.')[0] + '_vf_image.jpg'))

    centercrop(outDir)
    fivecrop(outDir)
    flip(outDir)


'''处理75/85张的数据集 扩容625倍'''


def augmentation_625(outDir):
    centercrop(outDir)
    fivecrop(outDir)


if __name__ == '__main__':
    outDir21 = r"C:\Users\Cooper Luo\Desktop\数据增强\data\黑色煤21"
    # outDir30 = r"C:\Users\Cooper Luo\Desktop\数据增强\data\灰黑色泥岩30"
    # outDir46 = r"C:\Users\Cooper Luo\Desktop\数据增强\data\灰色泥质粉砂岩46"
    # outDir18 = r"C:\Users\Cooper Luo\Desktop\数据增强\data\灰色细砂岩18"
    # outDir85 = r"C:\Users\Cooper Luo\Desktop\数据增强\data\浅灰色细砂岩85"
    # outDir40 = r"C:\Users\Cooper Luo\Desktop\数据增强\data\深灰色粉砂质泥岩40"
    outDir75 = r"C:\Users\Cooper Luo\Desktop\数据增强\data\深灰色泥岩75"
    # augmentation_2500(outDir18)
    augmentation_2500(outDir21)
    # augmentation_1875(outDir30)
    # augmentation_1875(outDir40)
    # augmentation_1250(outDir46)
    augmentation_625(outDir75)
    # augmentation_625(outDir85)
