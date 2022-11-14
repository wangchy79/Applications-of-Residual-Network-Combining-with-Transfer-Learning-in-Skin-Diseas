#### 使用训练好的模型请如下操作：

1. 保证dataset文件夹目录结构如下：

   ![image-20220624142340654](C:\Users\12785\AppData\Roaming\Typora\typora-user-images\image-20220624142340654.png)

   其中test文件夹中有对应测试图片

2. 保证`config.py`文件中`train = 0`，输入`python main.py`直接测试模型，其中`main.py`中有关测试有两种方法，具体可看`main.py`中最后一段代码注释。

#### 要训练后再运行代码，请按如下步骤：

1. 使用tensorflow和opencv-python

2. 保证dataset文件夹为空

3. 将原始数据放入original_dataset,如下所示：

   <img src="C:\Users\12785\AppData\Roaming\Typora\typora-user-images\image-20220624135109683.png" alt="image-20220624135109683" style="zoom: 33%;" />

4. 输入`python ./prepare_data`对所有原始数据进行增强

5. 配置config.py文件，如下所示：

   ```python
   # some training parameters
   EPOCHS = 5
   BATCH_SIZE = 80
   NUM_CLASSES = 40
   
   image_height = 168
   image_width = 168
   channels = 3
   save_model_dir = "saved_model/"
   dataset_dir = "dataset/"
   original_dir = "./original_dataset/"
   train_dir = dataset_dir + "train"
   valid_dir = dataset_dir + "valid"
   test_dir = dataset_dir + "test"
   
   Train_ratio=0.6
   Test_ratio=0.1
   
   # train or test? if train,train = 1
   train = 1
   
   # choose a network
   model = "ResNet152V2_Rahman"
   # model = "ResNet50_Mahbod"
   # model = "ResNet50_Hosseinzadeh"
   ```

   其中这里需要调整的参数如下：

   + Train_ratio指定训练集占总体数据的比例，这里为百分之六十
   + Test_ratio=0.1指定测试集，这里为百分之十，**剩下百分之三十为校验集**。

6. 输入`split_dataset`划分数据集为训练集，校验集，测试集。

7. 保证`config.py`文件中`train = 1`

8. 输入`python main.py`训练模型

9. 调整`config.py`文件中`train = 0`，输入`python main.py`测试模型，其中`main.py`中有关测试有两种方法，具体可看`main.py`中最后一段代码注释