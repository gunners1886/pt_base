import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader 
from torchinfo import summary
from tqdm import tqdm
import os, sys
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.abspath(os.path.join((os.path.dirname(__file__)), '..', '..')), ''))
from my_proj_config.my_proj_config import PROJ_ROOT


#设置随机数种子,便于复现
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#使用cuDNN加速卷积运算
torch.backends.cudnn.benchmark = True

# 载入MNIST数据集
# 载入训练集
train_dataset = torchvision.datasets.MNIST( root="dataset/", train=True, transform=transforms.ToTensor(), download=True )
#载入测试集
test_dataset = torchvision.datasets.MNIST( root="dataset/", train=False, transform=transforms.ToTensor(), download=True )
#生成dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32,shuffle=False)


# 教师网络
class TeacherModel(nn.Module):
    def __init__(self,in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu=nn.ReLU()
        self.fcl = nn.Linear(784,1200)
        self.fc2 = nn.Linear(1200,1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout=nn.Dropout(p=0.5)

    def forward(self,x):
        x=x.view(-1,784)
        x= self.fcl(x)
        x= self.dropout(x)
        x= self.relu(x)
        x=self.fc2(x)
        x=self.dropout(x)
        x= self.relu(x)
        x= self.fc3(x)
        return x

def load_model(model, path):
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        model.to(device)
        return True
    else:
        return False

# 从头训练教师模型
def train_teacher(b_save=True, b_read_from_local=True):

    str_path = os.path.join(PROJ_ROOT, 'models\knowledge_distrill\model_teacher.pth')
    

    
    model = TeacherModel()
    model = model.to(device)
    summary(model)

    if load_model(model, str_path):
        print("模型已加载，无需重新训练。")
        return model
    else:
        print(f"从头训练教师模型....")




        

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 6
    for epoch in range(epochs):
        model.train()
        # 训练集上训练模型权重
        for data, targets in tqdm(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            #前向预测
            preds = model(data)
            loss = criterion(preds, targets)
            # 反向传播,优化权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #测试集上评估模型性能
        model.eval()
        num_correct = 0
        num_samples = 0

        with torch.no_grad():
            for x,y in test_loader:
                x=x.to(device)
                y=y.to(device)

                
                preds = model(x)
                predictions = preds.max(1).indices
                num_correct += (predictions== y).sum()
                num_samples += predictions.size(0)
            acc = (num_correct/num_samples).item()

        model.train()
        print('Epoch:{}\t Accuracy:{:.4f}'.format(epoch+1,acc))


    if b_save:
        
        torch.save(model.state_dict(), str_path)
    return model

            





# 学生模型

class StudentModel(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(StudentModel,self).__init__()
        self.relu=nn.ReLU()
        self.fcl = nn.Linear(784,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20, num_classes)


    def forward(self, x):
        x=x.view(-1,784)
        x= self.fcl(x)
        # x = self.dropout(x)
        x= self.relu(x)
        x= self.fc2(x)
        # x = self.dropout(x)
        x= self.relu(x)
        x = self.fc3(x)
        return x


# 从头训练学生模型
def train_student_scratch():
    print(f"从头训练学生模型....")
    model = StudentModel()
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    epochs = 3
    for epoch in range(epochs):
        model.train()
        #训练集上训练模型权重
        for data, targets in tqdm(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            # 前向预测
            preds = model(data)
            loss = criterion(preds, targets)
            # 反向传播,优化权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        #测试集上评估模型性能
        model.eval()
        num_correct = 0
        num_samples = 0

        with torch.no_grad():
            for x,y in test_loader:
                x=x.to(device)
                y=y.to(device)


                preds = model(x)
                predictions = preds.max(1).indices
                num_correct += (predictions== y).sum()
                num_samples += predictions.size(0)
            acc = (num_correct/num_samples).item()

        model.train()
        print('Epoch:{}\t Accuracy:{:.4f}'.format(epoch+1,acc))
    
    return model





# 知识蒸馏训练学生模型
def train_student_kd(teacher_model):
    print(f"知识蒸馏训练学生模型....")
    #准备预训练好的教师模型
    teacher_model.eval()
    #准备新的学生模型
    model = StudentModel()
    model= model.to(device)
    model.train()


    #蒸馏温度
    temp = 7

    # hard loss
    hard_loss = nn. CrossEntropyLoss()
    # hard_loss 权重
    alpha=1.0


    # soft_loss
    soft_loss = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    epochs = 3
    for epoch in range(epochs):
        #训练集上训练模型权重
        for data, targets in tqdm(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            #教师模型预测
            with torch.no_grad():
                teacher_preds = teacher_model(data)
            #学生模型预测
            student_preds = model(data)
            #计算hard_loss
            student_loss = hard_loss(student_preds, targets)
            #计算蒸馏后的预测结果及soft_loss
            ditillation_loss = soft_loss(
                F.softmax(student_preds / temp, dim = 1),  
                F.softmax(teacher_preds / temp, dim = 1))
            #将hard_loss和soft_loss加权求和
            loss = alpha * student_loss + (1 - alpha) * ditillation_loss
            #反向传播,优化权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 测试集上评估模型性能
        model.eval()
        num_correct = 0
        num_samples = 0

        with torch.no_grad():
            for x,y in test_loader:
                x = x.to(device)
                y = y.to(device)
                preds =model(x)
                predictions = preds.max(1).indices
                num_correct += (predictions== y).sum()
                num_samples += predictions.size(0)
            acc = (num_correct/num_samples).item()

        model.train()
        print('Epoch:{}\t Accuracy:{:.4f}'.format(epoch+1,acc))

    
    return model




teacher_model = train_teacher()
student_model_scratch = train_student_scratch()
student_model_kd = train_student_kd(teacher_model)