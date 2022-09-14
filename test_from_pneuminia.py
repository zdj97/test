import torch
from torchvision import transforms, datasets
import json
import os
from mynet01 import MyNet
# from resmodel import resnet50, resnet101,res34
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from classic_net import classic_ResidualBlock01,classic_ResidualBlock,classic_ResidualBlock02,Bottleneck

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity",'F1_score']
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3)
            Recall = round(TP / (TP + FN), 3)
            Specificity = round(TN / (TN + FP), 3)
            F1_score=round((2*Precision*Recall)/(Precision+Recall),3)
            table.add_row([self.labels[i], Precision, Recall, Specificity,F1_score])
            #print('the model F1_score is ',F1_score)
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


# if __name__ == '__main__':

def test(image_path,net,mode_weight_path,num_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=2),

        transforms.ToTensor()])

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path ='I:\\肺炎检测实验数据\\input\\chest-xray-pneumonia\\'              #data_root + "/data_set/flower_data/"  # flower data set path
    image_path=image_path
    validate_dataset = datasets.ImageFolder(root=image_path + "test",
                                            transform=data_transform)

    batch_size = 1
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    net = net
    # load pretrain weights
    # model_weight_path = "I:\\肺炎\\代码\\Test5_resnet\\res50\\resNet50.pth"
    model_weight_path=mode_weight_path
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=num_class, labels=labels)
    net.eval()
    output_pro=[]
    lable=[]
    preds=[]
    fi=open('output.txt','w')
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            lable+=val_labels
            outputs = torch.softmax(outputs, dim=1)
            output_pro += outputs
            outputs = torch.argmax(outputs, dim=1)
            preds+=outputs
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    fi.write('output_pro: '+' '.join(map(str,output_pro)))
    fi.write('\n')
    fi.write('lable: '+' '.join(map(str,lable)))
    fi.write('\n')
    fi.write('preds: '+' '.join(map(str,preds)))

    confusion.plot()
    confusion.summary()

