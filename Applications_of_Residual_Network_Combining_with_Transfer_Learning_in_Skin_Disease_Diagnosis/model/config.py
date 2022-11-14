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
train = 0

# choose a network
model = "ResNet152V2_Rahman"
# model = "ResNet50_Mahbod"
# model = "ResNet50_Hosseinzadeh"