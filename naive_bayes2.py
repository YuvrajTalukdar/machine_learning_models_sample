from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import os
import pickle

def read_data(data_dir,data_div):
    all_data=[]
    all_label=[]
    file_list=os.listdir(data_dir)
    for file in file_list:
        for line in open(data_dir+file):
            if line.find("?")!=-1:
                continue
            line=line[:-1] #removing the new line symbol
            row=line.split(',')
            row=[float(i) for i in row]
            all_label.append(row[len(row)-1])
            row.pop(0)#to remove first element
            all_data.append(row[:-1])

    train_data,test_data,train_label,test_label=train_test_split(all_data,all_label,test_size=data_div,random_state=42)
    label_u=[]
    for label in all_label:
        #print(data)
        found=False
        for l in label_u:
            if l==label:
                found=True
                break
        if found!=True:
            label_u.append(label)

    print("train_data:",len(train_data))
    print("train_label: ",len(train_label))
    print("test_data: ",len(test_data))
    print("test_label: ",len(test_label))

    return train_data,train_label,test_data,test_label,len(label_u),label_u

gnb = GaussianNB()

def calc_total_accuracy(data,label):
    correct=0
    a=0
    test_label_predict = gnb.predict(data)
    while a<len(test_label_predict):
        if test_label_predict[a]==label[a]:
            correct+=1
        a+=1
    return (correct/len(data))*100

def calc_avg_accuracy(data,label):
    correct=0
    a=0
    test_label_predict = gnb.predict(data)
    while a<len(test_label_predict):
        if test_label_predict[a]==label:
            correct+=1
        a+=1
    return (correct/len(data))*100

train_data,train_label,test_data,test_label,no_of_classes,label_u=read_data("../processed_data_2/",0.5)#less than 0.997,

gnb.fit(train_data,train_label)

print("testing....")
print("total accuracy: ",calc_total_accuracy(test_data,test_label))

print("testing for each label")
f_test_data=[]
for l in label_u:
    test_data_per_label=[]
    a=0
    while a<len(test_data):
        if l==test_label[a]:
            test_data_per_label.append(test_data[a])
        a+=1
    f_test_data.append(test_data_per_label)

#print("total: ",len(test_data))
a=0
while a<len(f_test_data):
    #print(len(f_test_data[a]))
    print("accuracy for label",int(label_u[a]),": ",calc_avg_accuracy(f_test_data[a],label_u[a]))
    a+=1


pickle.dump(gnb,open("naive_bayes_model.sav",'wb'))
