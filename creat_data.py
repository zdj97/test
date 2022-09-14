
## 进行HIXIAOWEN训练集、验证集和测试集的制作
with open('/Users/zmj1997/Downloads/文档/KWS/代码/wekws/examples/hi_xiaowen'
          '/s0/data/train/data.list','r') as f:
    lines = f.readlines ()
out=[]
for line in lines:
    print(line)
    s=line.split('txt')
    S=s[1][3]
    n=line.split('"wav": ')
    if S=='-':
        S+=str(1)
    print(S)
    print(n[1][1:-3])
    t=S+' '+n[1][1:-3]
    out.append(t)
with open('/Users/zmj1997/Downloads/BC-ResNet/BC-ResNet/HIXIAOWEN/train_list.txt','w')as f:
    for o in out:
        f.writelines(o)
        f.writelines('\n')
