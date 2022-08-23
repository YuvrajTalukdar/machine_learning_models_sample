from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split

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

    return train_data,train_label,test_data,test_label,len(label_u)

train_data,train_label,test_data,test_label,no_of_classes=read_data("../processed_data_2/",0.5)

'''
count=0
count1=0
a=0
while a< len(train_data):
    if 2415==len(train_data[a]):
        count+=1
    elif 2416==len(train_data[a]):
        count1+=1
        print(train_label[a])
    a+=1
print(count)
print(count1)
val = input("Enter your value: ")
'''

rfc = RandomForestClassifier()
rfc.fit(train_data,train_label)
test_label_predict = rfc.predict(test_data)

print("testing....")
correct=0
a=0
while a<len(test_label_predict):
    if test_label_predict[a]==test_label[a]:
        correct+=1
    a+=1
print("accuracy: ",(correct/len(test_label))*100)