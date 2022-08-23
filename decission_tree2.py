from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import pickle

def read_data(data_dir,data_div):
    all_data=[]
    all_label=[]
    for line in open(data_dir):
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

    #print("train_data:",len(train_data))
    #print("train_label: ",len(train_label))
    #print("test_data: ",len(test_data))
    #print("test_label: ",len(test_label))

    return train_data,train_label,test_data,test_label,label_u

def calc_total_accuracy(model,data,label):
    correct=0
    a=0
    test_label_predict = model.predict(data)
    while a<len(test_label_predict):
        if test_label_predict[a]==label[a]:
            correct+=1
        a+=1
    return (correct/len(data))*100

def calc_avg_accuracy(model,data,label):
    correct=0
    a=0
    test_label_predict = model.predict(data)
    while a<len(test_label_predict):
        if test_label_predict[a]==label:
            correct+=1
        a+=1
    return (correct/len(data))*100

def train_and_test(dataset_dir,display_each,model):
    train_data,train_label,test_data,test_label,label_u=read_data("../processed_data_2/"+dataset_dir,0.2)#here 0.2 is percentage amount of test data
    print("train_data_len: ",len(train_data))
    print("test_data: ",len(test_data))
    model.fit(train_data,train_label)

    #print("testing....")
    result_file=open("result_file.txt",'w')
    total_accuracy=calc_total_accuracy(model,test_data,test_label)
    print("\n",dataset_dir," total accuracy: ",total_accuracy)
    result_file.write("\ntotal_accuracy: "+str(total_accuracy)+"\n")
    if display_each==True:
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
            label_accuracy=calc_avg_accuracy(model,f_test_data[a],label_u[a])
            result_file.write("\naccuracy for label"+str(int(label_u[a]))+": "+str(label_accuracy))
            print("accuracy for label",int(label_u[a]),": ",label_accuracy)
            a+=1
    result_file.close()
    pickle.dump(model,open("decission_tree_model.sav",'wb'))

model = DecisionTreeClassifier()
#model = SVC()
#model = RandomForestClassifier()
#model = GaussianNB()
#model = KNeighborsClassifier()
#model = LogisticRegression()
file_list=os.listdir("../processed_data_2/")
for file in file_list:
    train_and_test(file,True,model)
