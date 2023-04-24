#tensorboard --logdir=./run
#跑步出來:
#1.用絕對路徑
#2.換port 8088
#tensorboard --logdir=C:\Users\CNN\Downloads\ShauYuYan\S\runs_0419_rp --port 8088
#conda remove --name new --all
#--------------------------------------------------------
#                       import
#--------------------------------------------------------

import torch
import os , csv, json
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import csv
import h5py
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import warnings
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torch,torchmetrics
from torchmetrics.functional.classification import multiclass_auroc
torch.set_num_threads(1)
#conda create --name v2 python=3.9

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np 



class CBMIR():
    '''
    train + get_Feature+ retrieve

    NetWork有:
       DenseNet -> 'densenet'
       Vision Transformer  -> 'vit'
       Swin Transformer -> 'swin_vit'
       用list的方式輸入。

    auto_train()可以自動訓練。
    retrieve() 可以自動檢索。
    train_model() 用來做單次訓練。    
    '''
    def __init__(self,repeat_times = 0):
  
        self.data_path = " input_data\\" 
        self.save_path  = "save"
        self.model_listt=['swin_vit','vit','densenet']
        self.path_listt=['Contour','Hole','Organ']
        self.train_typee = ['finetune','trainFromScratch']
        self.num_epochs =5
        self.batch_size = 16
        self.top_list = [10]
        self.K = int
        self.repeat_times = repeat_times
        self.split_82 = True

    def train_model(self,model_name = str,num_epochs = int,batch_size = int ,pretrain = bool ,train_path = str,test_path = str,save_path= str,fold= int):
        def model_choose(model_name=str,num_classes=int,pretrain=bool):
            '''
            宣告model 
            NetWork有:
                DenseNet -> 'densenet'
                Vision Transformer  -> 'vit'
                Swin Transformer -> 'swin_vit'
                用str的方式輸入。
            '''
            def vit_model(num_classes=int,pretrain=bool):
                # Download pretrained ViT weights and model
                if pretrain:
                    vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT # "DEFAULT" means best available
                else:
                    vit_weights = False

                pretrained_vit = torchvision.models.vit_b_16(weights=vit_weights)
                
                # Freeze all layers in pretrained ViT model 
                for param in pretrained_vit.parameters():
                    param.requires_grad = True
               
                # Update the preatrained ViT head 
                embedding_dim = 768 # ViT_Base 16*16*3=768
                pretrained_vit.heads = nn.Linear(in_features=embedding_dim,out_features=num_classes)
                    
                model = pretrained_vit
                return model
        

            def swin_model(num_classes=int,pretrain=False):
                #載入預訓練參數
                if pretrain:
                    vit_weights = torchvision.models.Swin_B_Weights.DEFAULT
                    vit_weights = torchvision.models.Swin_V2_B_Weights
                else:
                    vit_weights = None
                
                print(vit_weights)
                #model = torchvision.models.swin_b(weights=vit_weights)
                model = torchvision.models.swin_v2_b(weights=vit_weights)
                #torchvision.models.swin_b(weights=torchvision.models.Swin_B_Weights.DEFAULT)
                #begin = nn.Sequential(*list(model.features.children())[0:4])
                # Freeze all layers in pretrained ViT model 
                for param in model.parameters():
                    param.requires_grad = True

                embedding_dim = 1024 # 根據SWIN_VIT版本不同而改變 S-> 768 B->1024
                model.head = nn.Linear(in_features=embedding_dim,out_features=num_classes)
                
                return model
        

            def densenet201(num_classes,pretrain=False):
                densenet201 = torchvision.models.densenet201(pretrained=pretrain)
                for param in densenet201.parameters():
                        param.requires_grad = True
                densenet201.classifier = nn.Linear(1920, num_classes)
                # densenet201.relu = nn.ReLU(inplace = True)
                # densenet201.fc2 = nn.Linear(1920, num_classes)
                #densenet201 = torch.load('ForBuddy_ori\\finetune\Hole\densenet\\3.pth')
                return densenet201
            
            
            def efficientnet(num_classes,pretrain=bool):
                #weights = torchvision.models.efficientnet_v2_s
                efficientnet = torchvision.models.efficientnet_v2_l(weights=pretrain)
                efficientnet.classifier = nn.Sequential(
                        nn.Dropout(p=0.3, inplace=True),
                        nn.Linear(1280, num_classes),
                    )
                print(efficientnet)
                #print("vcbncvbncvbncvbnvbn")
                return efficientnet
        
            #global model
            if model_name == "vit":
                model = vit_model(num_classes, pretrain)
            elif model_name == "swin_vit":
                model = swin_model(num_classes, pretrain)
            elif model_name == "densenet":
                model = densenet201(num_classes, pretrain)
            elif model_name == "efficientnet":
                model = efficientnet(num_classes, pretrain)
            #print(model)
            return model


        def train(train_loader, model, criterion, epoch, num_epochs, batch_size):
            model.train()
            total_train = 0
            correct_train = 0
            train_loss = 0

            lr = 0.01 #* (1/2)

            optimizer = optim.SGD(model.parameters(), lr= lr)
            # from torch.optim.lr_scheduler import StepLR
            # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            # 定义损失函数和优化器

            #optimizer = optim.Adam(model.parameters(),lr= lr)
            #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

            count = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                model.train()
                data = Variable(data)
                target = Variable(target)
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # clear gradient
                optimizer.zero_grad()

                # Forward propagation
                output = model(data) 
                loss = criterion(output, target) 

                # Calculate gradients
                loss.backward()
                
                # Update parameters
                optimizer.step()

                predicted = torch.max(output.data, 1)[1]
                total_train += len(target)
                correct_train += sum((predicted == target).float())
                train_loss += loss.item()

                if batch_idx % 1 == 0:
                    print("Train Epoch: {}/{} [iter： {}/{}], acc： {:.6f}, loss： {:.6f}".format(
                    epoch+1, num_epochs, batch_idx+1, len(train_loader),
                    correct_train / float((batch_idx + 1) * batch_size),
                    train_loss / float((batch_idx + 1) * batch_size)))
                count+=1
                if count == 29999:#5000
                    break
            #scheduler.step()        
            train_acc_ = 100 * (correct_train / float(total_train))

            # print(train_acc_.item(),type(train_acc_.item()))
            # print(a)
            train_loss_ = train_loss / total_train
                            
            return train_acc_.item(), train_loss_


        def validate(valid_loader, model, criterion, epoch, num_epochs, batch_size): 
            roc_predicted = None
            roc_target = None
            model.eval()
            total_valid = 0
            correct_valid = 0
            valid_loss = 0
            total_auc = 0
            total_specificity=0
            total_sensitivity=0
            total_ppv=0
            total_npv=0
            top1_correct = 0
            top5_correct = 0
            y_true = []
            y_scores = []
            count = 0
            for batch_idx, (data, target) in enumerate(valid_loader):
                #count += 1
                model.eval()
                data, target = Variable(data), Variable(target) 
                
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                output = model(data)
                loss = criterion(output, target) 

                predict = torch.softmax(output, dim=1) 
                predict = torch.max(output.data, 1)[1]

                total_valid += len(target)
                correct_valid += sum((predict == target).float())

                
                y_true.extend(target.tolist())
                y_scores.extend(predict.tolist())
                
                valid_loss += loss.item()

                if batch_idx % 1 == 0:
                    count+=1
                    print("Valid Epoch: {}/{} [iter： {}/{}], acc： {:.6f}, loss： {:.6f}".format(
                    epoch+1, num_epochs, batch_idx+1, len(valid_loader),
                    correct_valid / float((batch_idx + 1) * batch_size),
                    valid_loss / float((batch_idx + 1) * batch_size)))
                
                ##########################################################################
                # AUC compute
                ##########################################################################
                '''
                https://torchmetrics.readthedocs.io/en/stable/classification/auroc.html
                #from torchmetrics.functional.classification import multiclass_auroc
                '''
                auc = multiclass_auroc(output, target, num_classes=num_classes, average="macro", thresholds=None)
                total_auc += auc
                # print(auc)
                # from sklearn.metrics import roc_auc_score
                # auc = roc_auc_score(target.cpu().numpy(), predicted.cpu().numpy(),multi_class='ovo')
                # print("AUC:", auc)


                ##########################################################################
                # Sspecificity compute
                ##########################################################################
                '''
                https://torchmetrics.readthedocs.io/en/stable/classification/sspecificity.html
                '''
                #from torchmetrics.classification import MulticlassSspecificity
                from torchmetrics.classification import MulticlassSpecificity
                mcs = MulticlassSpecificity(num_classes=num_classes,average="micro").to(device)
                a = mcs(predict, target)
                total_specificity += a
                #print(a)
                #print('Sspecificity : ',a)

                
                ##########################################################################
                # sensitivity
                ##########################################################################
                from sklearn.metrics import recall_score
                from torchmetrics import classification
                #sensitivity = classification.F1Score(predict, target, average='multiclass')
                recall = classification.MulticlassRecall(num_classes) 
                sensitivity =  recall(predict.cpu(), target.cpu())
                #sensitivity = recall_score(target.cpu().numpy(), predict.cpu().numpy(),average='macro')
                #print(sensitivity)
                #print(asdasd)
                total_sensitivity += sensitivity


                ##########################################################################
                # PPV NPV
                ##########################################################################
                # import sklearn.metrics as metrics
                # #print((target.cpu().numpy()))
                # a = torch.tensor(((target.cpu().numpy())))
                # b = torch.tensor(((predict.cpu().numpy())))
                # #print(a,b)
                # #cm = metrics.confusion_matrix((a),predicted.cpu().numpy())
                # # 计算PPV和NPV
                # true_positives = (b * a).sum()
                # predicted_positives = b.sum()
                # PPV = true_positives / predicted_positives
                # # PPV = cm[1, 1] / (cm[1, 1] + cm[0, 1])
                # true_negatives = ((1-a) * (1-b)).sum()
                # predicted_negatives = (1-b).sum()
                # NPV =  true_negatives / predicted_negatives
                # NPV = cm[0, 0] / (cm[0, 0] + cm[1, 0])
                #(predict.cpu(), target.cpu())
                precision = torchmetrics.functional.classification.multiclass_precision(predict.cpu(), target.cpu(), num_classes) 
                recall = torchmetrics.functional.classification.multiclass_recall(predict.cpu(), target.cpu(), num_classes) 
                total_ppv +=  torch.tensor(precision)#PPV
                total_npv += torch.tensor(recall)#NPV
               # print("PPV:", precision,recall)
                #print(aaaaa)
                # #print(count)
                if count >= 29999:#2000
                    break
            #print(total_auc,count,float(total_auc)/count)
            #assert False
            valid_acc_ = 100 * (correct_valid / float(total_valid))
            valid_loss_ = valid_loss / total_valid
            total_auc = total_auc/count
            total_specificity = total_specificity / count
            total_sensitivity = total_sensitivity / count
            total_ppv = total_ppv / count
            total_npv = total_npv/ count
            #print(str(float(total_sensitivity))[:5])
            total_auc = 100 *float("{:.2f}".format(total_auc))
            total_specificity =100 *float( "{:.2f}".format(total_specificity))
            total_sensitivity = 100 *float("{:.2f}".format(total_sensitivity))
            total_ppv = 100 *float("{:.2f}".format(total_ppv))
            total_npv =100 *float( "{:.2f}".format(total_npv))

    
            #print(top5_correct/self.batch_size)
            #assert False
            #print(roc_target.shape,roc_predicted.shape)
            from sklearn.metrics import roc_curve, auc
            #fpr, tpr, thresholds = roc_curve(y_true= roc_target.cpu(), y_score= roc_predicted.cpu())
            try:
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                
                # 绘制 ROC 曲线
                import matplotlib.pyplot as plt
                plt.figure()
                lw = 2
                plt.plot(fpr, tpr, color='darkorange',
                        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic')
                plt.legend(loc="lower right")
            #plt.show()
                if not os.path.exists(save_path + '\\' + str(fold)):
                    os.makedirs(save_path + '\\' + str(fold))
                plt.savefig(save_path + '\\' + str(fold) + '\\'+str(epoch)+'_ROC.png')
                plt.clf()
                # print(fpr, tpr)
                # print(roc_auc)
            except:
                print('end')
            #assert False
            return valid_acc_.item(), valid_loss_,total_auc,total_specificity,total_sensitivity,total_ppv,total_npv


        def training_loop(model, criterion, train_loader, valid_loader, seconds, csv_name, num_epochs, batch_size,fold,save_path):
            # set objects for storing metrics
            maxx = 0
            ccount = 0
            total_train_loss = []
            total_valid_loss = []
            total_train_accuracy = []
            total_valid_accuracy = []
            training_accuracy = []
            valid_accuracy = []
            auc = []
            specificity = []
            sensitivity = []
            ppv = []
            npv = []
            CUDA = torch.cuda.is_available()
            device = torch.device("cuda" if CUDA else "cpu")
            # Train model
            for epoch in range(num_epochs):
                # training
                train_acc_, train_loss_ = train(train_loader, model, criterion, epoch, num_epochs, batch_size)
                total_train_loss.append(train_loss_)
                total_train_accuracy.append(train_acc_)
                formatted = "{:.2f}".format(train_acc_)
                training_accuracy.append(float(formatted))
                


                # validation
                with torch.no_grad():
                    valid_acc_, valid_loss_,total_auc,total_specificity,total_sensitivity,total_ppv,total_npv= validate(valid_loader , model, criterion, epoch,num_epochs, batch_size)
                    #print(valid_acc_, valid_loss_,total_auc,total_specificity,total_sensitivity,total_ppv,total_npv)
                    #
                    # print(total_auc)
                    # assert False
                    # ##自動停止
                    # if maxx == 0:
                    #     maxx =valid_acc_
                    # elif valid_acc_ > maxx:
                    #     maxx = valid_acc_
                    # else :
                    #     ccount = ccount + 1
                    # if ccount == 3:
                    #     return total_train_loss, total_valid_loss, total_train_accuracy, total_valid_accuracy

                    total_valid_loss.append(valid_loss_)
                    total_valid_accuracy.append(valid_acc_)
                    formatted = "{:.2f}".format(valid_acc_)
                    valid_accuracy.append(float(formatted))
                    auc.append(total_auc)
                    specificity.append(total_specificity)
                    sensitivity.append(total_sensitivity)
                    ppv.append(total_ppv)
                    npv.append(total_npv)

                #算時間
                now_time = int(time.time()-seconds)
                hr = 0
                mi = 0
                sec = 0
                while(now_time>3600):
                    now_time = now_time - 3600
                    hr = hr + 1
                while(now_time>60):
                    now_time = now_time - 60
                    mi = mi + 1
                sec = now_time
                if hr < 10:
                    hr = "0"+str(hr)
                else:
                    hr = str(hr)
                if mi < 10:
                    mi = "0"+str(mi)
                else:
                    mi = str(mi)
                if sec < 10:
                    sec = "0"+str(sec)
                else:
                    sec = str(sec)
                cost_time = hr + ":" + mi + ":" +sec


                print('================================================================================================================================')
                print("Epoch: {}/{}， Train acc： {:.6f}， Train loss： {:.6f}， Valid acc： {:.6f}， Valid loss： {:.6f}， Time： {}".format(
                    epoch+1, num_epochs, 
                    train_acc_, train_loss_,
                    valid_acc_, valid_loss_,cost_time))
                print('================================================================================================================================')
                #寫入訓練資料
                print(epoch,num_epochs)
                if epoch == num_epochs-1:
                    # with open(csv_name, 'a+', newline='') as csvfile:
                    #     writer = csv.writer (csvfile)
                    #     writer.writerow(["fold", "num_epochs", 
                    #         (str("train_acc_")),
                    #         (str("train_loss_")),
                    #         (str("valid_acc_")),
                    #         (str("valid_loss_")),"specificity","sensitivity","auc",
                    #         "cost_time"])
                    with open(csv_name, 'a+', newline='') as csvfile:
                        writer = csv.writer (csvfile)
                        writer.writerow([fold, num_epochs, 
                            (str(train_acc_)),
                            (str(train_loss_))[:7],
                            (str(valid_acc_)),
                            (str(valid_loss_))[:7],specificity[-1],sensitivity[-1],auc[-1],
                            cost_time])

            print("====== END ==========")
            print(training_accuracy,valid_accuracy,auc,specificity,sensitivity,npv,ppv)

            #assert False#total_sensitivity,total_ppv,total_npv
            # 畫出訓練準確率和測試準確率的折線圖
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
            #print(training_accuracy,valid_accuracy)
            plt.plot(range(1, num_epochs+1),training_accuracy, label='Train Accuracy')
            plt.plot(range(1, num_epochs+1),valid_accuracy, label='Test Accuracy')
            plt.plot(range(1, num_epochs+1),auc, label='auc')
            plt.plot(range(1, num_epochs+1),specificity, label='specificity')
            plt.plot(range(1, num_epochs+1),sensitivity, label='sensitivity')
            plt.plot(range(1, num_epochs+1),ppv, label='ppv')
            plt.plot(range(1, num_epochs+1),npv, label='npv')
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            # 設定 X 軸標籤
            plt.xlabel('Epoch')
            # 設定 Y 軸標籤
            plt.ylabel('Accuracy')
            # 設定圖形標題
            plt.title('Train and Test Accuracy')
            # 加入圖例
            plt.legend()
            # 顯示圖形
            #plt.show()
            if not os.path.exists(save_path + '\\' + str(fold)):
                os.makedirs(save_path + '\\' + str(fold))
            plt.savefig(save_path + '\\' + str(fold) + '\\figure.png')
            plt.clf()
                        #assert False#total_sensitivity,total_ppv,total_npv
            # 畫出訓練準確率和測試準確率的折線圖
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
            plt.plot(range(1, num_epochs+1),training_accuracy, label='Train Accuracy')
            plt.plot(range(1, num_epochs+1),valid_accuracy, label='Test Accuracy')
            plt.plot(range(1, num_epochs+1),auc, label='auc')
            plt.plot(range(1, num_epochs+1),specificity, label='specificity')
            plt.plot(range(1, num_epochs+1),sensitivity, label='sensitivity')
            # plt.plot(range(1, num_epochs+1),ppv, label='ppv')
            # plt.plot(range(1, num_epochs+1),npv, label='npv')
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            # 設定 X 軸標籤
            plt.xlabel('Epoch')
            # 設定 Y 軸標籤
            plt.ylabel('Accuracy')
            # 設定圖形標題
            plt.title('Train and Test Accuracy')
            # 加入圖例
            plt.legend()
            # 顯示圖形
            #plt.show()
            if not os.path.exists(save_path + '\\' + str(fold)):
                os.makedirs(save_path + '\\' + str(fold))
            plt.savefig(save_path + '\\' + str(fold) + '\\figure2.png')
            plt.clf()
            plt.close('all')
            #assert False
            print( training_accuracy[-1], valid_accuracy[-1])
            print( type(train_acc_), type(valid_acc_)) 
            return total_train_loss, total_valid_loss, training_accuracy, valid_accuracy,specificity,sensitivity,ppv,npv,auc

        #assert os.path.exists(train_path), "dataset train_path: {} does not exist.".format(train_path)
        # 遍历文件夹，一个文件夹对应一个类别
        flower_class = [cla for cla in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, cla))]
        print(flower_class)
        # 排序，保证顺序一致
        flower_class.sort()
        # 生成类别名称以及对应的数字索引
        class_indices = dict((k, v) for v, k in enumerate(flower_class))
        
        
        json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)


        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
        ]) # test transform

        dataset = datasets.ImageFolder(train_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        dataset = datasets.ImageFolder(test_path, transform=transform)
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        num_classes = (len(class_indices))


        warnings.filterwarnings('ignore')

        CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if CUDA else "cpu")
        

        model =model_choose(model_name, num_classes, pretrain)
        if CUDA:
            model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        seconds = time.time()


        #詳細訓練資料路徑
        csv_name =save_path + '\\'+'output.csv'

        #print("use vit_b_16 : ")
        print("model_name : ",model_name)
        print("device : ",device)
        print("train : ",(len(datasets.ImageFolder(train_path, transform=transform))))
        print("test  :  ",(len(datasets.ImageFolder(test_path, transform=transform))))
        print("num_classes = " + str(num_classes))
        print("pretrain = " + str(pretrain))
        print("num_epochs = " + str(num_epochs))

        #assert False
        total_train_loss, total_valid_loss, total_train_accuracya, total_valid_accuracyb,specificity,sensitivity,ppv,npv,auc = training_loop(model, criterion, train_loader, valid_loader,seconds,csv_name,num_epochs,batch_size,fold,save_path)
        #total_train_loss, total_valid_loss, training_accuracy, valid_accuracy,specificity,sensitivity,ppv,npv
        #print(a,b,c,d)
        #assert False
        print(time.ctime(seconds))
        torch.save(model, save_path + '\\'+str(fold)+".pth")
        # print(type(total_train_accuracy),total_train_accuracy)
        # print(type(total_valid_accuracy),total_valid_accuracy)
        a = total_train_accuracya
        b=  total_valid_accuracyb
        return a,b,specificity,sensitivity,ppv,npv,auc


    def auto_train(self):
        '''
        自動建立資料夾，自動訓練各種不同組合的model。
        '''
        # 使用默认参数创建summary writer，程序将会自动生成文件名
        writer = SummaryWriter()
        # 生成的文件路径为: runs_0419_rp5/Dec16_21-13-54_DESKTOP-E782FS1/

        # 创建summary writer时，指定文件路径
        writer = SummaryWriter("my_experiment")
        # 生成的文件路径为: my_experiment

        # 创建summary writer时，使用comment作为后缀
        writer = SummaryWriter(comment="LR_0.1_BATCH_16")

        writer = SummaryWriter('runs_0419_rp5')

        #----------------------------------------
        #----訓練前準備
        #----------------------------------------
        save_path = self.save_path
        for train_type in range(len(self.train_typee)):
            for model_list in range(len(self.model_listt)):
                for path_list in range(len(self.path_listt)):
                    for k in range(self.K):
                        save_pathh = save_path + '\\'+ self.train_typee[train_type]+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list]
                        if not os.path.exists(save_pathh):
                            os.makedirs(save_pathh)

                                            
        with open(self.save_path + '\output.csv', 'a+', newline='') as csvfile:
            csv.writer(csvfile).writerow(["train_type",
                                          "path_list",
                                          "model_list",
                                          "avg_train_acc",
                                          "avg_test_acc","avg_specificity",
                                          "avg_sensitivity",
                                          "avg_auc",
                                          "cost_time"])
                                       
        #----------------------------------------
        #-----開始訓練
        #----------------------------------------
        for train_type in range(len(self.train_typee)):
            for model_list in range(len(self.model_listt)):
                for path_list in range(len(self.path_listt)):
                    avg_train_acc = 0
                    avg_test_acc = 0
                    avg_specificity = 0
                    avg_sensitivity = 0
                    avg_auc = 0 

                    #計時開始
                    seconds = time.time()
                    for k in range(self.K):
                        # pretrain = True if self.train_typee[train_type] =='finetune' else False
        
                        # save_pathh = save_path + '\\'+ self.train_typee[train_type]+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list]
                        # train_path = 'input_data\\' +self.path_listt[path_list] +'\\fold' + str(k) + '\\train'
                        test_path = 'input_data\\' +self.path_listt[path_list] +'\\fold' + str(k) + '\\test' 
                        
                        # total_train_accuracy, total_valid_accuracy,specificity,sensitivity,ppv,npv,auc = self.train_model(self.model_listt[model_list],self.num_epochs,self.batch_size,pretrain,train_path,test_path,save_pathh,k)
                        # avg_train_acc += float(total_train_accuracy[-1])/(self.K)
                        # avg_test_acc += float(total_valid_accuracy[-1])/(self.K)
                        # avg_specificity += float(specificity[-1])/(self.K)
                        # avg_sensitivity += float(sensitivity[-1])/(self.K)
                        # avg_auc+= float(auc[-1])/(self.K)
                        
                        #輸出機率
                        self.probability_compute(test_path,save_pathh+'\\'+str(k)+'.pth',save_pathh+'\\'+str(k)+'\\'+str(k)+'_output.csv')
                       
                       
                       #繪製混淆矩陣
                        self.draw_confusion_atrix(test_path,
                                            save_pathh+'\\'+str(k)+'.pth'
                                            ,save_pathh+'\\'+str(k)+'\\'+str(k))   
                         
                        #資料可視化
                        writer = SummaryWriter('runs_0419_rp5/'+str(self.train_typee[train_type])
                                                  +'_'+str(self.model_listt[model_list])
                                                  +'_'+str(self.path_listt[path_list])
                                                  +'_'+str(self.K)
                                                  +'_'+str(self.repeat_times)
                                                )
                        continue
                        for n_iter in range(self.num_epochs):
                            writer.add_scalar('auc/auc_fold'+str(k),float(auc[n_iter]), n_iter)
                            writer.add_scalar('npv/npv_fold'+str(k),float(npv[n_iter]), n_iter)
                            writer.add_scalar('ppv/ppv_fold'+str(k),float(ppv[n_iter]), n_iter)
                            writer.add_scalar('sensitivity/sensitivity_fold'+str(k),float(sensitivity[n_iter]), n_iter)
                            writer.add_scalar('specificity/specificity_fold'+str(k), float(specificity[n_iter]), n_iter)
                            writer.add_scalar('train_Accuracy/train_fold'+str(k), float(total_train_accuracy[n_iter]), n_iter)
                            writer.add_scalar('test_Accuracy/test_fold'+str(k),float(total_valid_accuracy[n_iter]), n_iter)
                        writer.close()
                    
                    #計時結束
                    now_time = int(time.time()-seconds)
                    hr = 0
                    mi = 0
                    sec = 0
                    while(now_time>3600):
                        now_time = now_time - 3600
                        hr = hr + 1
                    while(now_time>60):
                        now_time = now_time - 60
                        mi = mi + 1
                    sec = now_time
                    if hr < 10:
                        hr = "0"+str(hr)
                    else:
                        hr = str(hr)
                    if mi < 10:
                        mi = "0"+str(mi)
                    else:
                        mi = str(mi)
                    if sec < 10:
                        sec = "0"+str(sec)
                    else:
                        sec = str(sec)
                    cost_time = hr + ":" + mi + ":" +sec
                    
                    #寫入詳細資料
                    with open(self.save_path + '\output.csv', 'a+', newline='') as csvfile:
                         csv.writer(csvfile).writerow([self.train_typee[train_type],
                                                        self.path_listt[path_list],
                                                        self.model_listt[model_list],
                                                        round(avg_train_acc,2),
                                                        round(avg_test_acc,2),
                                                        round(avg_specificity,2),
                                                        round(avg_sensitivity,2),
                                                        round(avg_auc,2),
                                                        (cost_time)])
                       
            
    def retireve(self):
        '''
        自動建立資料夾，自動訓練各種不同組合的model。
        '''    
        self.top_list.sort()
        for train_type in range(len(self.train_typee)):
            for model_list in range(len(self.model_listt)):
                for path_list in range(len(self.path_listt)):  
                    for top in range(len(self.top_list)):
                        avg_mAP = 0.0
                        import time                        
                        seconds = time.time()
                        timee = ''
                        for k in range(self.K):
                            if self.train_typee[train_type] == "trainFromScratch":
                                save_pathh = self.save_path + '\\retrieve_trainFromScratch'+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\fold' + str(k) 
                                #continue
                            else:
                                save_pathh = self.save_path + '\\finetune_retrieve'+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\fold' + str(k)
                            if not os.path.exists(save_pathh):
                                os.makedirs(save_pathh)
                            model_path = self.save_path + '\\'+ self.train_typee[train_type]+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\'+str(k)+'.pth'
                            
                            target_path = "target_data\\" + self.path_listt[path_list]
                            query_path = "query_data\\" + self.path_listt[path_list]

                            import time
                            seconds = time.time() 
                            if not os.path.exists(save_pathh):
                                 os.makedirs(save_pathh)
                            mAP = 0.0
                            if self.model_listt[model_list] == 'vit':
                                    if not os.path.exists(save_pathh +'\\target_data.h5'): 
                                        self.vit_get_feature(target_path,save_pathh,model_path)
                                        self.vit_get_feature(query_path,save_pathh,model_path)
                            if self.model_listt[model_list] == 'swin_vit':
                                    if not os.path.exists(save_pathh +'\\target_data.h5'): 
                                        self.swin_get_feature(query_path,save_pathh,model_path)
                                        self.swin_get_feature(target_path,save_pathh,model_path)
                                        
                            #         self.swin_get_feature(data_path,save_pathh,model_path)
                            #         mAP =self.ap_compute(save_pathh+ '\\' +self.path_listt[path_list] +'.h5',save_pathh+'\output.csv',(int(self.top_list[top])))
                            if self.model_listt[model_list] == 'densenet':
                                    if not os.path.exists(save_pathh +'\\target_data.h5'): 
                                        self.densenet_get_feature(target_path,save_pathh,model_path)
                                        self.densenet_get_feature(query_path,save_pathh,model_path)
                            #continue
                            
                            ap_path = save_pathh + '\\top_' + str(self.top_list[top])
                            if not os.path.exists(ap_path):
                                os.makedirs(ap_path)

                            #continue
                            mAP = self.ap_compute(save_pathh +'\\query_data.h5',save_pathh +'\\target_data.h5',ap_path+'\output.csv',(int(self.top_list[top])))
                            avg_mAP += mAP / int(self.K)
                          
                            now_time = int(time.time()-seconds)
                            hr = 0
                            mi = 0
                            sec = 0
                            while(now_time>3600):
                                now_time = now_time - 3600
                                hr = hr + 1
                            while(now_time>60):
                                now_time = now_time - 60
                                mi = mi + 1
                            sec = now_time
                            if hr < 10:
                                hr = "0"+str(hr)
                            else:
                                hr = str(hr)
                            if mi < 10:
                                mi = "0"+str(mi)
                            else:
                                mi = str(mi)
                            if sec < 10:
                                sec = "0"+str(sec)
                            else:
                                sec = str(sec)
                            cost_time = hr + ":" + mi + ":" +sec
                            with open(save_pathh+'\output.csv', 'a+', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([cost_time])
                            timee = cost_time
                        #now_time = int(time.time()-seconds)
                        with open(self.save_path + '\mAP_output.csv', 'a+', newline='') as csvfile:
                            csv.writer(csvfile).writerow([self.train_typee[train_type],
                                                            self.path_listt[path_list],
                                                            self.model_listt[model_list],
                                                            (int(self.top_list[top])),
                                                            avg_mAP,timee])

           
    def pretrain_retireve(self):
        '''
        自動建立資料夾，自動訓練各種不同組合的model。

        '''    
        self.top_list.sort()
        for train_type in range(len(self.train_typee)):
            for model_list in range(len(self.model_listt)):
                for path_list in range(len(self.path_listt)):  
                    for top in range(len(self.top_list)):
                        avg_mAP = 0.0
                        import time                        
                        seconds = time.time()
                        timee = ''
                        for k in range(self.K):
                            if self.train_typee[train_type] == "trainFromScratch":
                                save_pathh = self.save_path + '\\retrieve_trainFromScratch'+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\fold' + str(k) 
                                #continue
                            else:
                                save_pathh = self.save_path + '\\finetune_retrieve'+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\fold' + str(k)
                            if not os.path.exists(save_pathh):
                                os.makedirs(save_pathh)
                            model_path = self.save_path + '\\'+ self.train_typee[train_type]+'\\'+self.path_listt[path_list] + '\\'+self.model_listt[model_list] + '\\'+str(k)+'.pth'
                            
                            target_path = "target_data\\" + self.path_listt[path_list]
                            query_path = "query_data\\" + self.path_listt[path_list]

                            import time
                            seconds = time.time() 
                            if not os.path.exists(save_pathh):
                                 os.makedirs(save_pathh)
                            mAP = 0.0
                            if self.model_listt[model_list] == 'vit':
                                    if not os.path.exists(save_pathh +'\\target_data.h5'): 
                                        self.vit_get_feature(target_path,save_pathh,'')
                                        self.vit_get_feature(query_path,save_pathh,'')

                            if self.model_listt[model_list] == 'swin_vit':
                                    if not os.path.exists(save_pathh +'\\target_data.h5'): 
                                        self.swin_get_feature(query_path,save_pathh,'')
                                        self.swin_get_feature(target_path,save_pathh,'')
                                        
                            if self.model_listt[model_list] == 'densenet':
                                    if not os.path.exists(save_pathh +'\\target_data.h5'): 
                                        self.densenet_get_feature(target_path,save_pathh,'')
                                        self.densenet_get_feature(query_path,save_pathh,'')
                            #continue
                            
                            ap_path = save_pathh + '\\top_' + str(self.top_list[top])
                            if not os.path.exists(ap_path):
                                os.makedirs(ap_path)
                            mAP = self.ap_compute(save_pathh +'\\query_data.h5',save_pathh +'\\target_data.h5',ap_path+'\output.csv',(int(self.top_list[top])))
                            avg_mAP += mAP / int(self.K)
                          
                            now_time = int(time.time()-seconds)
                            hr = 0
                            mi = 0
                            sec = 0
                            while(now_time>3600):
                                now_time = now_time - 3600
                                hr = hr + 1
                            while(now_time>60):
                                now_time = now_time - 60
                                mi = mi + 1
                            sec = now_time
                            if hr < 10:
                                hr = "0"+str(hr)
                            else:
                                hr = str(hr)
                            if mi < 10:
                                mi = "0"+str(mi)
                            else:
                                mi = str(mi)
                            if sec < 10:
                                sec = "0"+str(sec)
                            else:
                                sec = str(sec)
                            cost_time = hr + ":" + mi + ":" +sec
                            with open(save_pathh+'\output.csv', 'a+', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([cost_time])
                            timee = cost_time
                        #now_time = int(time.time()-seconds)
                        with open(self.save_path + '\mAP_output.csv', 'a+', newline='') as csvfile:
                            csv.writer(csvfile).writerow([self.train_typee[train_type],
                                                            self.path_listt[path_list],
                                                            self.model_listt[model_list],
                                                            (int(self.top_list[top])),
                                                            avg_mAP,timee])


    def vit_get_feature(self,pred_path,save_folder,model_path):
           
        def vit_feature(img_path):
            '''
            extract feature from an image
            '''
            # image ->(3,224,224)
            transform = transforms.Compose([
                    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
            ])

            # use path open image
            img = Image.open(img_path)
            img = transform(img)

            #通道多一條
            img = torch.unsqueeze(img, dim=0)
            
            CUDA = torch.cuda.is_available()
            device = torch.device("cuda" if CUDA else "cpu")
            img = img.to(device)
            #print(img.shape)

            # create model
            #model = torch.load(model_path)
            model.eval()
            
            
            with torch.no_grad():
                x = model._process_input(img)
                n = x.shape[0]
                batch_class_token = model.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = model.encoder(x)
                x = x[:, 0]
                featuree = x.view(768).cpu().numpy()
                print(featuree.shape)
                return featuree

        '''
        extract feature from all image
        '''
        img_list = []
        feature_list = []
        if model_path =='':
            model = torchvision.models.vit_b_16(weights=1)
        else:
            model = torch.load(model_path)
        model.cuda()
        #計算圖片數量
        summ = 0
        for file_class in os.listdir(pred_path):
            for filename in os.listdir(pred_path +"\\"+ file_class):
                summ += 1

        #計算已處理圖片數量
        count = 0
        import numpy as np
        c = []
        for file_class in os.listdir(pred_path):
            for filename in os.listdir(pred_path +"\\"+ file_class):
                img_path = pred_path+"\\"+file_class+"\\"+filename
                img_list.append(img_path)
                img_feature = vit_feature(img_path)
                feature_list.append(img_feature)
                c.append(np.append(file_class,img_feature))

                count += 1
                print(str(pred_path).split("\\")[-1]+' : '+str(count)+'/'+str(summ))
                
            #break
            
                # 打开 CSV 文件，指定写入模式
        with open('output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            # 写入表头
            writer.writerow(['col' + str(i+1) for i in range((768))] + ['label'])

            # 写入数据
            for row in c:
                writer.writerow(row)     
                    
        #return 0
        #write in h5

        with h5py.File(save_folder+'\\'+str(pred_path).split("\\")[0]+'.h5', 'w') as f:
            f.create_dataset("path", data=img_list)
            f.create_dataset("feature", data=feature_list)

        #check h5
        with h5py.File(save_folder+'\\'+str(pred_path).split("\\")[0]+'.h5', "r") as k:
            print( k.keys())
            print(k.get('path').size)
            print(k.get('feature').size/(k.get('path').size))


    def swin_get_feature(self,pred_path,save_folder,model_path):
            
            def swin_feature(img_path,model_path):


                transform = transforms.Compose([
                        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
                ]) 

                img = Image.open(img_path)
                img = transform(img)
                img = torch.unsqueeze(img, dim=0)
                CUDA = torch.cuda.is_available()
                device = torch.device("cuda" if CUDA else "cpu")
                img = img.to(device)

                # create model
                
                model.eval()


                with torch.no_grad():
                    #https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029/2
                    input = model.features(img)
                    #7,7,1024 -> 1,1,1024
                    avgPooll = nn.AdaptiveAvgPool2d(1)
                    input = torch.transpose(input, 1, 3)#把通道维放到最后
                    output = avgPooll(input)

                    #swin b 1024 features 1,1,1024-> 1024
                    featuree = output.view(1024).cpu().numpy()
                    return featuree

            if model_path == '':
                model = torchvision.models.swin_v2_b(1)
            else:
                model = torch.load(model_path)

            img_list = []
            feature_list = []
            
            #計算圖片數量
            summ = 0
            for file_class in os.listdir(pred_path):
                for filename in os.listdir(pred_path +"\\"+ file_class):
                    summ += 1

            #計算已處理圖片數量
            count = 0
            for file_class in os.listdir(pred_path):
                for filename in os.listdir(pred_path +"\\"+ file_class):
                    img_path = pred_path+"\\"+file_class+"\\"+filename
                    img_list.append(img_path)
                    feature_list.append(swin_feature(img_path,model_path))
                    count += 1
                    print(str(pred_path).split("\\")[-1]+' : '+str(count)+'/'+str(summ))
                
            #write in h5
            with h5py.File(save_folder+'\\'+str(pred_path).split("\\")[0]+'.h5', 'w') as f:
                f.create_dataset("path", data=img_list)
                f.create_dataset("feature", data=feature_list)

            #check h5
            with h5py.File(save_folder+'\\'+str(pred_path).split("\\")[0]+'.h5', "r") as k:
                print( k.keys())
                print(k.get('path').size)
                print(k.get('feature').size/(k.get('path').size))


    def densenet_get_feature(self,pred_path,save_folder,model_path):
        # model = torch.load(model_path)
        def densenet_feature(img_path,model_path):


            transform = transforms.Compose([
                    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
            ]) 

            img = Image.open(img_path)
            img = transform(img)
            img = torch.unsqueeze(img, dim=0)
            CUDA = torch.cuda.is_available()
            device = torch.device("cuda" if CUDA else "cpu")
            img = img.to(device)

            # create model
            
           
            a = model.features.children
            

            with torch.no_grad():
                input = model.features(img)
                avgPooll = nn.AdaptiveAvgPool2d(1)

                output = avgPooll(input)
                output = torch.transpose(output, 1, 3)#把通道维放到最后
                featuree = output.view(1920).cpu().numpy()
                print(featuree.shape)
                return featuree

        img_list = []
        feature_list = []
        if model_path =='':
                 model = torchvision.models.densenet201(pretrained=1)
        else:
                model = torch.load(model_path)
        model.cuda()
        model.eval()
        #計算圖片數量
        summ = 0
        for file_class in os.listdir(pred_path):
            for filename in os.listdir(pred_path +"\\"+ file_class):
                summ += 1

        #計算已處理圖片數量
        count = 0
        c=[]
        import numpy as np
        for file_class in os.listdir(pred_path):
            for filename in os.listdir(pred_path +"\\"+ file_class):
                img_path = pred_path+"\\"+file_class+"\\"+filename
                img_list.append(img_path)
                ff = densenet_feature(img_path,model_path)
                feature_list.append(ff)
                c.append(np.append(file_class,ff))
                count += 1
                print(str(pred_path).split("\\")[-1]+' : '+str(count)+'/'+str(summ))
                with h5py.File(save_folder+'\\'+str(pred_path).split("\\")[-1]+'.h5', 'w') as f:
                    f.create_dataset("path", data=img_list)
                    f.create_dataset("feature", data=feature_list)
                #break
            #break
        with open('output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            # 写入表头
            writer.writerow(['col' + str(i+1) for i in range((1920))] + ['label'])

            # 写入数据
            for row in c:
                writer.writerow(row)         
            

        #write in h5
        with h5py.File(save_folder+'\\'+str(pred_path).split("\\")[0]+'.h5', 'w') as f:
            f.create_dataset("path", data=img_list)
            f.create_dataset("feature", data=feature_list)

        #check h5
        with h5py.File(save_folder+'\\'+str(pred_path).split("\\")[0]+'.h5', "r") as k:
            print( k.keys())
            print(k.get('path').size)
            print(k.get('feature').size/(k.get('path').size))


    def ap_compute(self,query_path = str,target_path = str,save_path = str,top = int):
       
        def mAP_output(path = str):
            num_classes = {}
            classes = {}
            mAP = 0
            with open(path, newline='') as csvfile:
                next(csv.reader(csvfile))
                for row in csv.reader(csvfile):
                    try:
                        roww = str(row[1])
                    except:
                        continue
                    if str(row[1]) not in classes:
                        classes.update({str(row[1]):float(row[0])})
                        num_classes.update({str(row[1]):1})
                    else:
                        classes.update({str(row[1]):(classes[str(row[1])]*num_classes[str(row[1])]+float(row[0]))/(1+num_classes[str(row[1])])})
                        num_classes.update({str(row[1]):num_classes[str(row[1])]+1})
                for i in classes:
                    mAP += classes[i] / len(classes)
                return(mAP,classes)


        top = top

        #----------------------------------------------------------
        #              input feature and path
        #----------------------------------------------------------
        import time
        seconds = time.time()
        feature_list = []
        feature_path = []
        query_list = []
        query_path_list = []
        with h5py.File(target_path, "r") as k:
            for i in range(len(k.get('feature'))):
                feature_list.append(k.get('feature')[i])
                feature_path.append(k.get('path')[i])
        with h5py.File(query_path, "r") as k:
            for i in range(len(k.get('feature'))):
                query_list.append(k.get('feature')[i])
                query_path_list.append(k.get('path')[i])

        #----------------------------------------------------------
        #               mAP compute
        #----------------------------------------------------------

        #將結果寫入csv
        #先宣告要用的欄位
        # ap , 分類 , 路徑
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ap','category','path'])
        print(len(feature_list))
        #mAP init
        mAP = 0
        #開始跑每張圖
        for j in range(0,len(query_list)):
            #continue
            print(j)
        
            #查詢影像
            query = query_list[j]
            query_path = query_path_list[j]

            #用來存放每張圖的cosin similarity
            score_map = {}

            #比對資料庫中的每一筆DATA
            for i in range(len(feature_list)):
                #計算cosin similarity
                cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

                #需要轉tensor
                cosine_similarity = cos(torch.from_numpy(query),torch.from_numpy(feature_list[i]))
                #寫入 dic
                score_map.update({cosine_similarity:feature_path[i]})

            #將前 n 相似的輸出
            top_list = {}
            for i in range(top):
                #每次都挑最大的放入dic 概念類似選擇排序
                top_list.update({max(score_map):score_map[max(score_map)]})

                #將最大的刪除
                del score_map[max(score_map)]
            
            relevant = 0 
            #印出top-n串列
            for i in top_list:
                if (str(query_path).split('\\')[-3]) == (str(top_list[i]).split('\\')[-3]):
                    relevant = relevant + 1
                print(top_list[i])

            print(relevant,top)
            ap = relevant/top
            print(save_path.split("\\")[-3:-2])
            print(query_path)
            print('ap:',ap)
            mAP += ap/len(query_list)

            #write in csv 
            with open(save_path, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([ap,(str(query_path).split('\\')[-3]),str(query_path)])
                csvfile.close()
            #break
       
        
            #writer.writerow([cost_time])
        
        mAP,classes = mAP_output(save_path)
        print('mAP : ',mAP)
        with open(save_path, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in classes:
                    writer.writerow([row,classes[row]])
                writer.writerow(['mAP',mAP])
        return mAP


    def foldd(self,data_path = str):
        '''
        自動分資料
        '''
        import shutil
        import os
        import random
        # k = self.K 
        # print(k)
        # assert False
        def splite82(path=str,save_path=str):    
            '''
            input  : 欲分82資料夾路徑。
            output : 將資料夾以同病人同資料集的方式分82，分train & test 資料集。
            '''
        #----------------- -----------------------
        # 讀資料 & 宣告字典 & 建立串列
        #-----------------------------------------
            #用於計算單個病人的圖片數{病人A:5張照片 }
            CT_count = {}
            path = path# + '\\'
            #以下暫存變數
            img = '' #判斷是否同系列用
            all_count = 0 #計算圖片總數
            count = 0 #計算單個病人的圖片數

            for i in os.listdir(path):
                #判斷是否同系列
                if str(img) != (str(i)):
                    #print(img,str(i.split('_')[1]))
                    img = (str(i))
                    count = 1
                #將病人的圖片數寫入字典 ex.{病人A:5張照片 }
                CT_count.update({img:count})
                #print({img:count})
                count += 1
                all_count+=1


            ############################################
            #check 1
            count = 0
            for i in CT_count:
                count+=CT_count[i]
            print("############################################+\ncheck1 : ")
            print('as',count,all_count)
            ############################################


            train = []
            train_count = 0
            for i in CT_count:
                if self.split_82 == True:
                    if train_count < all_count * (0.8):
                        ran = random.randint(0,90)#設90剛好設99會有少分的狀況
                        if ran < 80:
                            train.append(i)
                            train_count += CT_count[i]
                else:
                    train.append(i)
                    train_count += CT_count[i]
                    
            #不在train 中即test 
            test = []
            for i in CT_count:
                if i not in train:
                    test.append(i)

            ############################################
            #check 2
            print("############################################+\ncheck2 : ")
            print(f'train  {train_count}  :  all  {all_count} : 80% == {(all_count * 0.8)}')
            print(f'performance : {(train_count /all_count)}')
            count = 0
            for i in train:
                count += CT_count[i]
            for i in test:
                count += CT_count[i]
            print(f'number check : {count}')
            ###########################################
        
            save_path_train = save_path +'\\train\\' 
            if not os.path.exists(save_path_train):
                os.makedirs(save_path_train)

            save_path_test = save_path + '\\test\\' 
            if not os.path.exists(save_path_test):
                os.makedirs(save_path_test)
            print("###########################################")
            print("分資料中....")
            
            #使用複製的方式，避免影響原始資料。
            for j in os.listdir(path):
                        #print(path + j)
                        if str(j)  in train:
                            shutil.copy((path+'\\'+j),(save_path_train + j))
                        if str(j)  in test:
                            shutil.copy((path+'\\'+j),(save_path_test+j))
            print("fin")


        #輸入,輸出
        for i in os.listdir(data_path +"\\"):
            for j in os.listdir("data\\" + i):
                print(i+'/'+j)
                if not os.path.exists('data2\\'+i+'\\'+j):
                    os.makedirs('data2\\'+i+'\\'+j)
                splite82('data\\'+i+'\\'+j,'data2\\'+i+'\\'+j)

        

        add_path = 'serise'+'\\train'
        if not os.path.exists(add_path):
            os.makedirs(add_path)
    

        count = 0
        for i in os.listdir('data2'):
            for j in os.listdir('data2\\' + i):
                for k in os.listdir('data2\\'+ i + '\\'+j ):
                    for n in os.listdir('data2\\'+ i + '\\'+j+ '\\'+k ):
                            if not os.path.exists('serise'+'\\'+k+'\\'+i+'\\'+j+'\\'):
                                os.makedirs('serise'+'\\'+k+'\\'+i+'\\'+j+'\\')
                            
                            if k =='train':
                                os.rename(('data2\\'+ i + '\\'+j+ '\\'+k +'\\'+n),('serise'+'\\train\\'+i+'\\'+j+'\\'+n)) 
                            else:
                                continue
                            count += 1
                            print(count,n)
                            # else:
                            #      os.rename(('data2\\'+ i + '\\'+j+ '\\'+k +'\\'+n),('serise'+'\\test\\'+i+'\\'+j+'\\'+n))

        def kfold(path=str,save_path=str,k = self.K ):    
            '''
            input  : 欲分82資料夾路徑。
            output : 將資料夾以同病人同資料集的方式分82，分train & test 資料集。
            '''
        #----------------- -----------------------
        # 1.讀資料 & 宣告字典 & 建立串列
        #-----------------------------------------
            #k = self.K 
            #用於計算單個病人的圖片數{病人A:5張照片 }
            CT_count = {}
            path = path + '\\'
            #以下暫存變數
            img = '' #判斷是否同系列用
            all_count = 0 #計算圖片總數
            count = 0 #計算單個病人的圖片數

            for i in os.listdir(path):
                #判斷是否同系列
                if str(img) != (str(i)):
                    #print(img,str(i.split('_')[1]))
                    img = (str(i))
                    count = 1
                #將病人的圖片數寫入字典 ex.{病人A:5張照片 }
                CT_count.update({img:count})
                #print({img:count})
                count += 1
                all_count+=1
        

            ############################################
            #check 1
            count = 0
            for i in CT_count:
                count+=CT_count[i]
            print("############################################+\ncheck1 : ")
            print('as',count,all_count)
            print('check1 end')
            print("############################################")
            ############################################
            
            #----------------- -----------------------
            # 2. k fold 分 test set
            #-----------------------------------------
            #{31043692: fold 1},
            kfold = {}
            train = []
            test_count = 0
            bb = 0
            performence = {}
            #sett = ''
            #分每一個set
            for i in range(k):
                #print(i)
                test_count = 0
                for j in CT_count:
                    #print(j,i)
                    if j in kfold:
                        continue
                    if test_count < all_count * ((1/k)*0.9):
                        #亂數設定，有些分train有些分test 
                        ran = random.randint(0,90)#設90剛好設99會有少分的狀況
                        if ran < 80:
                            sett = str(j)
                            kfold.update({sett:i})
                        #print({a:i})
                        test_count += CT_count[j]
                
                performence.update({i:test_count})
                print(test_count,i)
                bb += test_count
            #print(bb)
            for j in CT_count:
                    if j in kfold: 
                        continue
                    else :
                        kfold.update({j:k-1})
                        print(({j:k-1}))
                        bb +=CT_count[j] 
            #print(bb)
                        #print(test_count)
                    # print(i,CT_count[i])
            ############################################
            #check 2
            print("############################################+\ncheck2 : ")
            count = 0
            for i in kfold:
                print('serise : ',i,'count :',CT_count[i],"kflod set : ",kfold[i])
                count += CT_count[i]
                print(count)
            print('total : ',count)
            print('check2 end')
            print("############################################")
            ############################################

            #----------------- -----------------------
            # 3. k fold 分  dataset
            #-----------------------------------------
            for i in range(k):
                #continue
                for j in os.listdir(path):
                    serise = str(j)
                    if kfold[serise] == i :
                        shutil.copy((path+j),(save_path +'/fold'+str(i) +'/test/'+ str(j)))
                        continue    
                        #print(j,kfold[serise])
                    shutil.copy((path+j),(save_path +'/fold'+str(i) +'/train/'+ str(j)))
            print(save_path)
            
            ############################################
            # #check 3
            # count = 0 
            # for i in os.listdir(save_path):
            #     check = set()
            #     for j in os.listdir(save_path +'/'+ i):
            #         for n in os.listdir(save_path +'/'+ i+'/'+ j):
            #             if j == 'test':
            #                 check.add(str(n).split("_")[0])
            #         for n in os.listdir(save_path +'/'+ i+'/'+ j):
            #             if j == 'train':
            #                 if  str(n).split("_")[0] in check:
                                
            #                     assert False ,'分錯了' + str(n).split("_")[0]
            # print("pass")
            # print(performence)
                
            ############################################

        def createFold(save_path):
            #creat n fold folder
            #save_path = 'test'
            for i in range(self.K):
                #continue

                add_path = save_path +'\\fold'+str(i)+'\\train'
                if not os.path.exists(add_path):
                    os.makedirs(add_path)
                add_path = save_path +'\\fold'+str(i)+'\\test'
                if not os.path.exists(add_path):
                    os.makedirs(add_path)
            #kfold('data\Contour\gallbladder_Contour','test')

        for i in os.listdir("serise\\train"):
            for j in  os.listdir("serise\\train" + '\\'+i):
                    print("serise\\train" + '\\'+i+ '\\'+j + '\\')
                    add_path = 'ok_data\\'+ i + '\\'+j
                    if not os.path.exists(add_path):
                        os.makedirs(add_path)
                    createFold(add_path)
                    kfold("serise\\train" + '\\'+i+ '\\'+j,add_path,self.K )

        def createFold(save_path):
            #creat n fold folder
            #save_path = 'test'
            
                #continue
            for label in os.listdir("ok_data"):
                for organ in os.listdir("ok_data\\" + label):
                    for fold in  os.listdir("ok_data\\" + label + '\\'+ organ):
                        for traintest in os.listdir("ok_data\\" + label + '\\'+ organ + '\\' + fold ):
                            print(label,organ,fold)
                            add_path = save_path +'\\'+label + '\\' + fold + '\\' +traintest+'\\'+organ
                            if not os.path.exists(add_path):
                                        os.makedirs(add_path)
        

        #for i in os.listdir("ok_data"):
        createFold('input_data')
        #assert False
        all = 0
        for label in os.listdir("ok_data"):
            for organ in os.listdir("ok_data\\" + label):
                for fold in  os.listdir("ok_data\\" + label + '\\'+ organ):
                    for traintest in os.listdir("ok_data\\" + label + '\\'+ organ + '\\' + fold):
                        for img in os.listdir("ok_data\\" + label + '\\'+ organ + '\\' + fold + '\\' +traintest):
                            all +=1

        count = 0                    
        for label in os.listdir("ok_data"):
            for organ in os.listdir("ok_data\\" + label):
                for fold in  os.listdir("ok_data\\" + label + '\\'+ organ):
                    for traintest in os.listdir("ok_data\\" + label + '\\'+ organ + '\\' + fold):
                        for img in os.listdir("ok_data\\" + label + '\\'+ organ + '\\' + fold + '\\' +traintest):
                            count +=1
                            old  = ("ok_data\\" + label + '\\'+ organ + '\\' + fold + '\\' +traintest+'\\'+img)
                            new  = ("input_data\\" + label + '\\'+ fold + '\\' +  traintest+ '\\' +organ+'\\'+img)
                            os.rename(old,new)
                            print(count,'/',all)
                        #assert False
        import os,shutil

        #create query_data  
        for i in os.listdir('data2'):
            for j in os.listdir('data2\\' + i):
                for k in os.listdir('data2\\' + i + '\\' + j):
                    if k == 'train':
                            continue
                    for m in os.listdir('data2\\' + i + '\\' + j + '\\' + k):
                        print('data2\\' + i + '\\' + j + '\\' + k + '\\' + m)
                        
                        add_path = 'query_data\\' + i +'\\' + j 
                        if not os.path.exists(add_path):
                                        os.makedirs(add_path)
                        old  = ("data2\\" + i + '\\'+ j + '\\' + k + '\\' +m)
                        new  = add_path +'\\' + m
                        os.rename(old,new)
        #create target_data           
        # for i in os.listdir('input_data'):
        #     for j in os.listdir('input_data\\' + i):
        #         for k in os.listdir('input_data\\' + i + '\\' + j):
        #             for m in os.listdir('input_data\\' + i + '\\' + j + '\\' + k):
        #                 for n in os.listdir('input_data\\' + i + '\\' + j + '\\' + k+ '\\' + m):
        #                     old = ('input_data\\' + i + '\\' + j + '\\' + k+ '\\' + m + '\\'+ n)
        #                     new = ('target_data'  + '\\' + i+ '\\' + m + '\\'+ n)
        #                     if not os.path.exists('target_data'  + '\\' + i+ '\\' + m ):
        #                         os.makedirs('target_data'  + '\\' + i+ '\\' + m )
        #                     shutil.copy(old,new)
        # 删除非空目录
        shutil.rmtree('data2')
        shutil.rmtree('serise')
        shutil.rmtree('ok_data')

    
    def probability_compute(self,path,model_path,probability_save):

       # print(confusion_atrix_save)
        #assert Flase

        mapp = {}
        test_labels = []
        pred_labels = []
        classes = []

        def predict(path,model_path):   
            def img_predict(img_path,model):
                    root = path
                    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
                    # 遍历文件夹，一个文件夹对应一个类别
                    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
                    print(flower_class)
                    # 排序，保证顺序一致
                    flower_class.sort()
                    # 生成类别名称以及对应的数字索引
                    class_indices = dict((k, v) for v, k in enumerate(flower_class))
                    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
                    # with open('class_indices.json', 'w') as json_file:
                    #     json_file.write(json_str)
                    print('=====================================================================================')
                    print("img PATH : "+str(img_path))
                    print("model PATH : "+str(model_path))
                    print('=====================================================================================')
                    #warnings.filterwarnings('ignore')
                    CUDA = torch.cuda.is_available()
                    device = torch.device("cuda" if CUDA else "cpu")
                    #print(device)

                    transform = transforms.Compose([
                            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
                    ]) # test transform
                    img_path = img_path
                    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
                    img = Image.open(img_path)
                    
                    img = transform(img)
                    img = torch.unsqueeze(img, dim=0)
                    json_path = 'class_indices.json'
                    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
                
                    with open(json_path, "r") as f:
                        class_indict = json.load(f)
                    #model = torch.load(model_path)
                    model = model
                
                    model.eval()
                    with torch.no_grad():
                        # predict class
                        img = img.to(device)
                        output = torch.squeeze(model(img)).cpu()
                        predict = torch.softmax(output, dim=0) 
                        predict_cla = torch.argmax(predict).numpy()
                    
                    print(predict)
                    #assert False
                    probability = torch.max(predict)
                    print(predict_cla,  probability,predict)
                    print()
                    pred = (format(class_indict[str(predict_cla)]))
                    #red = mapp[pred]
                    label = (img_path.split('\\')[-2])
                    label = mapp[label]
                    test_labels.append(label)
                    pred_labels.append(pred)
                    
                    correct = 1 if str(img_path).split('\\')[-2]  == pred else 0
                    with open (probability_save, 'a+', newline='') as csvfile:
                        csv.writer(csvfile).writerow([img_path,pred,probability.item(),predict])
                        csvfile.close()
                    if correct == 0:
                        with open ('miss.csv', 'a+', newline='') as csvfile:
                            csv.writer(csvfile).writerow([img_path,pred,probability.item(),predict])
                            csvfile.close()
                    
                
                    return correct
                    
            
            model = torch.load(model_path)
            conut = 0
            correct = 0
            for i in os.listdir(path):
                    classes.append(i)
                    mapp.update({i:conut})
                    conut += 1
            conut = 0
            for i in os.listdir(path):
                for j in os.listdir(path + '\\'+i):
                    conut += 1
                    correct += img_predict(path + '\\'+i+'\\'+j,model)
            print(correct,conut)
            print(correct / conut)
            return 0
        predict(path,model_path)
     #
    
    def draw_confusion_atrix(self,path,model_path,confusion_atrix_save):
        # 設定隨機種子以確保可重複性
        torch.manual_seed(42)
        np.random.seed(42)
       # print(confusion_atrix_save)
        #assert Flase

        mapp = {}
        test_labels = []
        pred_labels = []
        classes = []

        def predict(path,model_path):   
            def img_predict(img_path,model):
                    root = path
                    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
                    # 遍历文件夹，一个文件夹对应一个类别
                    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
                    print(flower_class)
                    # 排序，保证顺序一致
                    flower_class.sort()
                    # 生成类别名称以及对应的数字索引
                    class_indices = dict((k, v) for v, k in enumerate(flower_class))
                    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
                    with open('class_indices.json', 'w') as json_file:
                        json_file.write(json_str)
                    print('=====================================================================================')
                    print("img PATH : "+str(img_path))
                    print("model PATH : "+str(model_path))
                    print('=====================================================================================')
                    #warnings.filterwarnings('ignore')
                    CUDA = torch.cuda.is_available()
                    device = torch.device("cuda" if CUDA else "cpu")
                    #print(device)

                    transform = transforms.Compose([
                            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
                    ]) # test transform
                    img_path = img_path
                    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
                    img = Image.open(img_path)
                    
                    img = transform(img)
                    img = torch.unsqueeze(img, dim=0)
                    json_path = 'class_indices.json'
                    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
                
                    with open(json_path, "r") as f:
                        class_indict = json.load(f)
                    #model = torch.load(model_path)
                    model = model
                
                    model.eval()
                    with torch.no_grad():
                        # predict class
                        img = img.to(device)
                        output = torch.squeeze(model(img)).cpu()
                        predict = torch.softmax(output, dim=0) 
                        predict_cla = torch.argmax(predict).numpy()

                    pred = (format(class_indict[str(predict_cla)]))
                    pred = mapp[pred]
                    label = (img_path.split('\\')[-2])
                    label = mapp[label]
                    test_labels.append(label)
                    pred_labels.append(pred)
                    # print(pred,label)
                    # print(pred_labels)
                    #assert False
                    return 0
                    
            
            model = torch.load(model_path)
            conut = 0
            for i in os.listdir(path):
                    classes.append(i)
                    mapp.update({i:conut})
                    conut += 1

            for i in os.listdir(path):
                for j in os.listdir(path + '//'+i):
                    img_predict(path + '\\'+i+'\\'+j,model)
                    
            return 0

        predict(path,model_path)


        # 計算混淆矩陣
        cm = confusion_matrix(test_labels, pred_labels, normalize='true')

        # 繪製混淆矩陣
        fig, ax = plt.subplots(figsize=((len(classes))+1, (len(classes))))
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        # 设置水平标签
        ax.set_xlabel('Predicted Class')
        # 设置垂直标签
        ax.set_ylabel('True Class')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        for iiii in range(len(classes)):
            for jjjj in range(len(classes)):
                print(jjjj,iiii)
                print((len(classes)))
                ax.text(jjjj, iiii, "{:.2f}".format(cm[iiii, jjjj]),
                        ha="center", va="center", color="white")
        plt.title("Confusion matrix")
        plt.colorbar(im)
        plt.savefig(confusion_atrix_save+'Confusion matrix.png')
        plt.clf()
        #plt.show()

#cut dataset
# cb = CBMIR()
# cb.K = 5
# cb.foldd('data')
# print(asd)
for i in range(1,2):
    cb = CBMIR(i)
    cb.data_path = " input_data\\" 
    cb.repeat_times = i
    cb.save_path="S_8_"+str(i)
    cb.train_typee = ['finetune']
    cb.model_listt=['vit']
    cb.path_listt=['S']
    cb.num_epochs =1
    cb.batch_size = 16
    cb.top_list = [10]
    cb.K = 5
    cb.probability_compute(path ='query_data\\S',model_path='S_8_1\\finetune\S\\vit\\0.pth', probability_save='result/prob.csv')
    # cb.auto_train()
    # cb.retireve()
#fin