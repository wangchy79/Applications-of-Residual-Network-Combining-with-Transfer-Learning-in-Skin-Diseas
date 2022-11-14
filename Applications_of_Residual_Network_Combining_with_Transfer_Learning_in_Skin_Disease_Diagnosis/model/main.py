from wsgiref import validate
from get_data import load_data,load_data_test
from tensorflow import keras
import config
import matplotlib.pyplot as plt
from models.model import ResNet152V2_Rahman,ResNet50_Hosseinzadeh,ResNet50_Mahbod


def get_model():
    if config.model == "ResNet152V2_Rahman":
        model = ResNet152V2_Rahman()
    elif config.model == "ResNet50_Hosseinzadeh":
        model = ResNet50_Hosseinzadeh()
    elif config.model == "ResNet50_Mahbod":
        model = ResNet50_Mahbod()
    return model

# 主程序
def main():

    if config.train == 1:
        # 调用keras的ResNet152V2模型
        model = get_model()

        # 给出训练和测试数据
        X_train, Y_train,X_valid,Y_valid, X_test, Y_test = load_data()
        print('X_train shape : ', X_train.shape)
        print('Y_train shape : ', Y_train.shape)
        print('X_test shape : ', X_test.shape)
        print('Y_test shape : ', Y_test.shape)

        # 训练模型
        training = model.fit(X_train, Y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,validation_data = (X_valid,Y_valid))

        # 把训练好的模型保存到文件
        model.save(config.save_model_dir + config.model)
    
        # 画图看一下训练的效果

        # accuracy
        fig = plt.figure(dpi=500)
        plt.plot(training.history['accuracy'])
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        fig.savefig("accuracy.png")

        # loss
        fig = plt.figure(dpi=500)
        plt.plot(training.history['loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        fig.savefig("loss.png")
        print("train success!")
    
    else:
        model = keras.models.load_model(config.save_model_dir + config.model)
        X_test, Y_test = load_data_test()

        print('X_test shape : ', X_test.shape)
        print('Y_test shape : ', Y_test.shape)

        # 评估模型,第一种方式，输出每张图片真实标签和对应预测标签
        # ppp = model.predict(X_test)
        # for i in range(len(X_test)):
        #     max_v = 0.0
        #     max_i = 0
        #     t_j = 0
        #     for j in range(len(ppp[i])):
        #         if ppp[i][j] > max_v:
        #             max_v = ppp[i][j]
        #             max_i = j
            
        #     for j in range(len(Y_test[i])):
        #         if Y_test[i][j] == 1.0:
        #             t_j = j
        #     print(t_j,"\t",max_i)
        
        #第二种方式，输出accuracy和loss
        model.evaluate(X_test, Y_test, batch_size=40)
 
if __name__ == "__main__":
    main()