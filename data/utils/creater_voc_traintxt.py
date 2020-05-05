import os
import random 
random.seed(1) 
xmlfilepath=r'/home/zj/OCR/projects/FasterRCNN/data/VOCdevkit2007/VOC2007/Annotations'
saveBasePath=r"/home/zj/OCR/projects/FasterRCNN/data/VOCdevkit2007/VOC2007"
 
trainval_percent=1.0#0.8
train_percent=0.8#0.7
total_xml = os.listdir(xmlfilepath)
num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size",tv)
print("traub suze",tr)
ftrainval = open(os.path.join(saveBasePath,'ImageSets/Main/trainval.txt'), 'w')  
#ftest = open(os.path.join(saveBasePath,'ImageSets/Main/test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'ImageSets/Main/train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'ImageSets/Main/val.txt'), 'w')  
 
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    #else:  
        #ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
#ftest .close()

