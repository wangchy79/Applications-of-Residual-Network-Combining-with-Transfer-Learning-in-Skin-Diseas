import pathlib
import config
import numpy as np
import cv2

def get_images_and_labels(data_root_dir):

    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]

    return all_image_path


'''opencv数据增强
    对图片进行色彩增强、高斯噪声、水平镜像、放大、旋转、剪切
'''


def contrast_brightness_image(src1, a, g, path_out):
    '''
        色彩增强（通过调节对比度和亮度）
    '''
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    # addWeighted函数说明:计算两个图像阵列的加权和
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)
    cv2.imwrite(path_out, dst)


def gasuss_noise(image, path_out_gasuss, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    cv2.imwrite(path_out_gasuss, out)


def mirror(image, path_out_mirror):
    '''
        水平镜像
    '''
    h_flip = cv2.flip(image, 1)
    cv2.imwrite(path_out_mirror, h_flip)

def rotate(image, path_out_rotate):
    '''
        旋转
    '''
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(path_out_rotate, dst)


def shear(image, path_out_shear):
    '''
        剪切
    '''
    height, width = image.shape[:2]
    cropped = image[int(height / 9):height, int(width / 9):width]
    cv2.imwrite(path_out_shear, cropped)


def Data_Augmentation(dataset_root_dir):
    
    imageNameList = [
        '_color.jpg',
        '_gasuss.jpg',
        '_mirror.jpg',
        '_rotate.jpg',
        '_shear.jpg']
    all_image_path= get_images_and_labels(data_root_dir=dataset_root_dir)
    print(all_image_path[0][:-4])
    for i in range(0, len(all_image_path)):
        path = all_image_path[i]
        for j in range(5):
            path_out = all_image_path[i][:-4]+ imageNameList[j]
            image = cv2.imread(path)
            if j == 0:
                contrast_brightness_image(image, 1.2, 10, path_out)
            elif j == 1:
                gasuss_noise(image, path_out)
            elif j == 2:
                mirror(image, path_out)
            elif j == 3:
                rotate(image, path_out)
            elif j == 4:
                shear(image, path_out)
            print(i,": "+ path_out + " success！")

def main():
    Data_Augmentation(dataset_root_dir=config.original_dir)#对所有数据做数据增强再划分
    #Data_Augmentation(dataset_root_dir=config.valid_dir)  
if __name__ == '__main__':
    main()