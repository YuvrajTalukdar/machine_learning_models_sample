import os
import pickle

def read_data(data_dir):
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
    f_data_arr=[]
    a=0
    while a<len(label_u):
        f_data=[]
        b=0
        while b<len(all_label):
            if label_u[a]==all_label[b]:
                f_data.append(all_data[b])
            b+=1
        f_data_arr.append(f_data)
        a+=1

    print("all_label:",len(all_label))
    print("all_data:",len(all_data))
    
    print("train_data:",len(all_label))
    print("train_label: ",len(all_data))
    
    return all_data,all_label,len(label_u),label_u

data,label,_,_=read_data("../processed_data_2/")

gnb=pickle.load(open("naive_bayes_model.sav",'rb'))
dtc=pickle.load(open("decission_tree_model.sav",'rb'))

print("naive_bayes score: ",gnb.score(data,label))
print("decission_tree score: ",gnb.score(data,label))

correct=0
a=0
nb_result=gnb.predict(data)
while a<len(data):
    if label[a]==nb_result[a]:
        correct+=1
    a+=1
print("Total Accuracy Naive Bayes: ",(correct/len(data))*100)
correct=0
a=0
dtc_result=gnb.predict(data)
while a<len(data):
    if label[a]==dtc_result[a]:
        correct+=1
    a+=1
print("Total Accuracy Decission Tree: ",(correct/len(data))*100)
print(dtc_result)
