import torch
import get_data
import numpy as np
import torchaudio


def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()

def cal_FAR_FFR(pred,target):
    FA=FF=0
    FALSE=TRUE=0
    for i in range(pred.shape[0]):
        if target[i]==2:
            FALSE+=1
            if pred[i]==0 or pred[i]==1:
                FA+=1
        else:
            TRUE+=1
            if pred[i]==2:
                FF+=1
    #print('非唤醒词共{}个，唤醒词共{}个'.format(FALSE,TRUE))
    return FA, FF,FALSE,TRUE

def get_likely_index(tensor,out):
    index=tensor.argmax(dim=-1)
    # print(index)
    for i in range(index.shape[0]):
      #  print(index[i])
        if index[i].item()==1 or index[i].item()==0:
            if out[i][index[i]].item()<=8.9:
                index[i]=2
               # print(index[i])
    return index


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = fa = ff = 0
    total_non=total_wakeup=0
    out_pro=[]
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        pred, out = model(data)
#        for i in range(pred.shape[0]):
#            if target[i].item()==2 and (pred[i].item()==1 or pred[i].item()==0):
#                with open('fa误报的概率.txt','a+') as f:
#                    for i in range(out.shape[0]):
#                        f.writelines(' '.join(map(str,out[i].cpu().detach().numpy()))+'\n')
        with open('all_pro.txt','a+') as f:
            for i in range(out.shape[0]):
                f.writelines(' '.join(map(str,out[i].cpu().detach().numpy()))+'\n')
        X=torch.softmax(pred,dim=-1)
        Pred = get_likely_index(pred,out)  ## pred
        for i in range(Pred.shape[0]):
            if target[i].item()==2 and (Pred[i].item()==1 or Pred[i].item()==0):
                with open('fa_error.txt','a+') as f:
                    f.writelines(' '.join(map(str,out[i].cpu().detach().numpy()))+'\n')
        #print(pred)
       # out_pro+=[X[i][Pred[i]].item() for i in range(Pred.shape[0])]
        correct += number_of_correct(Pred, target)
        a,f,t_non,t_wake = cal_FAR_FFR(Pred,target)
        fa+=a
        ff+=f
        total_non+=t_non
        total_wakeup+=t_wake
    #print(PRED.shape)
    #with open('out_pro.txt','w') as f:
    #    for x in out_pro:
    #        f.writelines(str(x)+'\n')
    score = correct / len(data_loader.dataset)
    FAR = fa / 52177
    FFR = ff / (10642*2)
    print('测试数据数量：{}, 唤醒词共：{}，非唤醒词共{}, 一共误报了{} 个，漏报了{}个'.format(len(data_loader.dataset),total_wakeup,total_non,fa,ff))
    return score,FAR,FFR


def apply_to_wav(model, waveform: torch.Tensor, sample_rate: float, device: str):
    model.eval()
    mel_spec = get_data.prepare_wav(waveform, sample_rate)
    mel_spec = torch.unsqueeze(mel_spec, dim=0).to(device)
    res = model(mel_spec)

    probs = torch.nn.Softmax(dim=-1)(res).cpu().detach().numpy()
    predictions = []
    for idx in np.argsort(-probs):
        label = get_data.idx_to_label(idx)
        predictions.append((label, probs[idx]))
    return predictions


def apply_to_file(model, wav_file: str, device: str):
    waveform, sample_rate = torchaudio.load(wav_file)
    return apply_to_wav(model, waveform, sample_rate, device)
