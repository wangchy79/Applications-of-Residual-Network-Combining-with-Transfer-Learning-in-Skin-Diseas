import config
import pathlib
from config import image_height, image_width, channels
import numpy as np
from tensorflow.keras.preprocessing import image
import random

def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_dataset(dataset_root_dir):
    all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    # 准备训练数据集，ResNet50的缺省输入为224x224
    X = np.empty((len(all_image_path), image_height, image_width, channels))
    #此次分类任务类别为40
    Y = np.empty((len(all_image_path), 40))
    count = 0
    for img_path in all_image_path:
        img = image.load_img(img_path, target_size=(image_height, image_width))
        img = image.img_to_array(img) / 255.0
 
        # 训练的输入为图像，输出为分类，输出是一个one-hot向量
        X[count] = img
        Y[count] = np.zeros(40)
        Y[count][all_image_label[count]] = 1.0
        count += 1
        print(count)
    # print("image_path: {}".format(all_image_path[:]))
    # print("image_label: {}".format(all_image_label[:]))
    
    # 打乱顺序
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    Y = Y[index]

    return X, Y



def load_data():
    X_train,Y_train = get_dataset(dataset_root_dir=config.train_dir)
    X_valid,Y_valid = get_dataset(dataset_root_dir=config.valid_dir)
    X_test,Y_test = get_dataset(dataset_root_dir=config.test_dir)
    return X_train,Y_train,X_valid,Y_valid,X_test,Y_test

def load_data_test():
    X_test,Y_test = get_dataset(dataset_root_dir=config.test_dir)
    return X_test,Y_test
if __name__ == '__main__':
    X_train,Y_train,X_test,Y_test = load_data()
