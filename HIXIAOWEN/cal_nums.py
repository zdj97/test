num0=num1=num2=0

f=open('./testing_list.txt','r')
for line in f.readlines():
    l=line.split()
    if l[0]=='0':
        num0+=1
    elif l[0]=='1':
        num1+=1
    else:
        num2+=1
print(num0,num1,num2)
