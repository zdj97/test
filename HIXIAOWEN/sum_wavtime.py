import wave
import os
file_path='./testing_list.txt'
total_time=0
nonwaketime=waketime=0
f=open(file_path,'r') 
for line in f:
    label,file_name=line.split() 
    #print(label)
    with wave.open (file_name, 'rb') as f:
        frames = f.getnframes ()
        rate = f.getframerate ()
        wav_length = frames / float (rate)
        if label=='-1':
            nonwaketime+=wav_length
        else:
            waketime+=wav_length
        total_time+=wav_length
print('非唤醒词测试总时长为{},唤醒词测试总时长为{},测试总时长为{}'.format(nonwaketime,waketime,total_time))
