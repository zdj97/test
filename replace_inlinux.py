fi=open('/data/zdj/BC-ResNet_HIXIAOWEN/BC-ResNet/HIXIAOWEN/testing_list_bk.txt','r')
fo=open('/data/zdj/BC-ResNet_HIXIAOWEN/BC-ResNet/HIXIAOWEN/testing_list.txt','w')
lines=fi.readlines()
for line in lines:
    l=line.replace('/Users/zmj1997/fsdownload/HIXIAOWEN/data_tar','/data/zdj/wekws/examples/hi_xiaowen/s0/data/local')
    fo.writelines(l)
   # fo.writelines('\n')

fi.close()
fo.close()
