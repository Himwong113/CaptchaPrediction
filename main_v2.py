import torch
import torch.nn.functional as F
import tqdm
from DataSet import CaptchaDataset
from model_v2 import CapatchaModel
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import visdom
import os

traindataloader = DataLoader(dataset=CaptchaDataset(partition='train' ,dig=5,catergory=36  ),batch_size=5,shuffle=True)
validdataloader = DataLoader(dataset=CaptchaDataset(partition="valid" ,dig=5,catergory=36  ),batch_size=3,shuffle=True,drop_last=True)
testdataloader = DataLoader(dataset=CaptchaDataset(partition='test'   ,dig=5,catergory=36  ),batch_size=1,shuffle=True)
print(f'check GPU state :{torch.cuda.is_available()} \n True-> cuda else False-> cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs =2000
#model =CaptchaReg().to(device)
model =CapatchaModel(label_size=(5,36)).to(device)
opt = Adam(model.parameters(),lr=0.0001)
scheduler = CosineAnnealingLR(opt,T_max=epochs)
epoch_list = []
loss_list =[]
vis = visdom.Visdom(env='Captcha')
best_acc = 0
if not os.path.exists('output'):
    os.makedirs('output')
for epoch in range(epochs):
        #print(f'Now Run epoch {epoch}...')
        model.train()

        loss_sum = 0
        count = 0
        for idx ,(image,label) in enumerate( tqdm.tqdm(traindataloader)):
                #print(idx)
                #print(image.shape) #batch,channel,width,height
                #print(label.shape) #batch, char,charvalue

                if idx == 0 and epoch == 0:
                    print(f'size of dataset confirmation:\nimage:{image.shape},label:{label.shape} ')
                    model.label_size = label.shape[1:]
                    model.to(device)
                    print(
                        f'i.e. in output layer, it will softmax each {label.shape[1]} dig of {label.shape[2]} catergories ')
                image = image.to(device)
                label = label.to(device)
                opt.zero_grad()
                logit = model(image).to(device)
                #print(logit.shape)
                loss = F.cross_entropy(logit,label,reduction='mean')

                loss.backward()
                opt.step()
                loss_sum = loss_sum +loss.item()
                count = count+1
                #if idx %49 ==0 :
                #        print(f'ita :{idx},loss:{loss_sum/count}')

        print(f'epoch :{epoch},loss:{loss_sum/count},lr:{scheduler.get_last_lr()}')
        scheduler.step()
        epoch_list.append(epoch)
        loss_list.append(loss_sum/count)
        vis.scatter(torch.tensor([epoch]), torch.tensor([loss_sum/count]), win='Test Loss', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Test Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'})
        model.eval()
        acc = 0
        total_dig = 0
        valid_loss_sum = 0
        valid_count = 0
        for idx, (image, label) in enumerate(tqdm.tqdm(validdataloader)):
            image = image.to(device)
            label = label.to(device)
            #if idx == 0 and epoch == 0:
            #    print(f'size of dataset confirmation:\nimage:{image.shape},label:{label.shape} ')
            model.eval()
            logit = model(image).to(device)
            loss = F.cross_entropy(label, logit, reduction='mean')
            valid_loss_sum += loss.item()
            valid_count += 1
            _, logit_out = torch.max(logit, dim=-1)
            _, label_out = torch.max(label, dim=-1)
            if idx == 0:
                print(f'\nlogit   output = {logit_out}')
                print(f'GT      output = {label_out} ')

            acc += torch.sum(logit_out == label_out)
            total_dig += torch.numel(label_out)

        print(f'\nepoch={epoch + 1},train loss ={loss_sum / count},valid_loss ={valid_loss_sum / valid_count},',
              f'\nacc={acc / total_dig * 100}% where best acc ={best_acc}\n')
        epoch_acc =acc / total_dig * 100
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(),f'output/epoch_{epoch}.pt')
        vis.scatter(torch.tensor([epoch]), torch.tensor([valid_loss_sum / valid_count]), win='Valid Loss', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Test Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'})


model.eval()
print('Enter Eval state....')
for idx,(image,label) in enumerate(tqdm.tqdm(testdataloader)):
        logit = model(image) #batch ,6,10
        logit_max = torch.max(logit,dim=-1)[1] #batch ,6,1

        #print(logit_max)
        label_max = torch.max(label,dim=-1)[1]
        print(f'pred:{logit_max} , ture:{label_max}')