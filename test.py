import os

import torch
import torch.nn.functional as F
import tqdm
from DataSet import CaptchaDataset
from model_v2 import CapatchaModel
from torch.utils.data import DataLoader

testdataloader = DataLoader(dataset=CaptchaDataset(partition='test'   ,dig=5,catergory=36  ),batch_size=1,shuffle=True)
print(f'check GPU state :{torch.cuda.is_available()} \n True-> cuda else False-> cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model =CapatchaModel(label_size=(5,36)).to(device)


#torch.load()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'output')
model_list =os.listdir(DATA_DIR)
best_acc =0
best_epoch =""
for model_file in model_list:
    try:
        model.load_state_dict(torch.load(os.path.join( DATA_DIR,model_file)))

        model.eval()
        print('Enter Eval state....')
        valid_loss_sum =0
        valid_count =0
        acc=0
        total_dig =0
        epoch = model_file.split('_')[1]
        epoch = epoch.split('.')[0]
        for idx,(image,label) in enumerate(tqdm.tqdm(testdataloader)):
            image = image.to(device)
            label = label.to(device)
            # if idx == 0 and epoch == 0:
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
        print(f'acc={acc},dig={total_dig}')
        epoch_acc=acc / total_dig * 100
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_epoch = model_file
        print(f'\nepoch={epoch },valid_loss ={valid_loss_sum / valid_count},',
              f'\nacc={epoch_acc}% where best acc ={best_acc}% @ {best_epoch}\n')
        acc = 0
        total_dig = 0
    except:
        print(f'{model_file} is not for this model')
