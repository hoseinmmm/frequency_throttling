import pandas as pd
import joblib
import sys
import numpy as np
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import pickle
import datetime


def save_list_to_file(file_path='ts_data_list.pickle', data_list=[]):
    with open(file_path, 'wb') as file:
        pickle.dump(data_list, file)

def load_list_from_file(file_path='ts_data_list.pickle'):
    with open(file_path, 'rb') as file:
        data_list = pickle.load(file)
    return data_list


def prepare_dataset(path='',re_load=False,n_models=15,samples=10,fname=''):
    global family_based
    

    if re_load:
        
        ignore_model =['7', '6', '4', '10', '28'] # not good models

        #model = []
        instancet = [[0],[0],[0],[0],[0],[0]]
        instances_l = [instancet[:]]*(n_models*samples) # 15*40
        
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"File: {file_path}")
                if ("csv" in file_path) or (".DS_" in file_path):
                    continue
                
                data_list = str(file).strip().split('_')

                #print('data_list:', data_list)

                md = data_list[3]

                if md in ignore_model:
                    continue


                # since the name of model is not in order, we have to rename them to make them in order!
                if (int(md)>19 and int(md)<30):
                    # set 2 - unseen model
                    if int(md)>28:
                        md = str(int(md)-21)
                    else:
                        # set 2
                        md = str(int(md)-20)
            
                elif (int(md)>34):
                     md = str(int(md)-26)

                
                else:
                    # set 1
                    if int(md) >3:
                        if int(md)==5:
                            md = str(int(md)-2)
                        elif int(md)>7 and int(md)<10:
                            md = str(int(md)-4)
                        elif int(md)>10 and int(md)<20:
                            md = str(int(md)-5)
                        elif int(md)>29 and int(md)<35:
                            md = str(int(md)-15)
                    else:
                        md = str(int(md)-1)



                
                v_idx = int(data_list[5]) - 1
                element = int(data_list[7][:-4]) 



                
                '''
                # here I just mange samples, if your run completed without interrupt, do not change elements
                if fname=='_utest':
                    print('no skip elements')

                elif fname=='_ktest':
                    if element > 16:
                        element = element - 17
                    else:
                        continue
                elif element > 16:
                    continue
                '''
                

                #print('element:', element)    
                ins_idx = (int(md)*samples)+element  

                instance = instances_l[ins_idx][:]

                variable = []
                with open(file_path, 'r') as f:
                    for line in f:
                        val = int(line.rstrip())
                        # clipping outliers, it is not mandotory !
                        if val > 300:
                            val = 300
                        variable.append(str(val))


                instance[v_idx] = variable[:]
                instances_l[ins_idx] = instance[:]


                '''
                if ins_idx in [a for a in range(120,160,1)]:
                    
                    print(ins_idx, v_idx, md,element)
                    #print(variable[0])
                    print(instances_l[ins_idx][v_idx][0:10])


                    input()
                '''
                
                #print('instance:', instance)

                #input()
        #print("array after parse")
        #print(instances_l[0][0][0])
        #print(instances_l[0][5][0])
        #print(instances_l[160][0][0])
        #print(instances_l[160][5][0])

        print('nummber of samples in dataset: ',len(instances_l))

        save_list_to_file(file_path=f'instances_l{fname}.pickle', data_list=instances_l)
        print('nummber of samples in instances_l: ',len(instances_l))
        #save_list_to_file(file_path='model.pickle', data_list=model)
        #print('nummber of samples in model: ',len(model))


        return instances_l#, model
    
    else:
        print("loading pickl ...")
        instances_l = load_list_from_file(f'instances_l{fname}.pickle')
        #model = load_list_from_file('model.pickle')
        print('nummber of samples in instances_l: ',len(instances_l))
        #print('nummber of samples in model: ',len(model))
        return instances_l#, model





        


def get_family(m,test_type):

    if test_type=='u':
        # set 2
        family_dic = {   '0' : ['5'], # model 20 = ConvNext
            '1' : ['0','1','2', '3'], # 21 VIT
            '2' : ['6','7','4'], # 22 Resnet
            '3' : ['6','7','4'], # 23 resnet
            '4' : ['8','9','10'], # 24 mobilnet
            '5' : ['11','12'], # 25 inception
            '6' : ['13','14'], # 26 bert
            '7' : ['6','7','4'], # 27 resnet
            '8' : ['13','14'], # 29 bert
            '9' : ['15','16','17'], # 35 VGG
            '10' : ['18','19'], # 36  MIT
        }
    else:

        # set 1
        # 15 models and 6 families
        family_dic = {   '0' : ['0','1','2', '3'], #1 vit
            '1' : ['0','1','2', '3'], #vit 
            '2' : ['0','1','2', '3'], #vit 
            '3' : ['0','1','2', '3'], #vit 
            '4' : ['6','7','4'], # resnet
            '5' : ['5'], # convnext
            '6' : ['6','7','4'],  # resnet
            '7' : ['6','7','4'],  # resnet
            '8' : ['8','9','10'], # mobilnet
            '9' : ['8','9','10'],# mobilnet
            '10' : ['8','9','10'],# mobilnet
            '11' : ['11','12'], # inception
            '12' : ['11','12'], # inception
            '13' : ['13','14'], #  bert
            '14' : ['13','14'], #  bert
            '15' : ['15','16','17'], #  VGG
            '16' : ['15','16','17'], #  VGG
            '17' : ['15','16','17'], #  VGG
            '18' : ['18','19'], # 36  MIT
            '19' : ['18','19'], # 36  MIT
        }


    if m in family_dic:
        return family_dic[m]
    else:
        return [m] 



def min_max_normalize(arr,min_value,max_value):

    normalized_arr = (arr - min_value) / (max_value - min_value)
    return normalized_arr


def z_score_normalize(arr):
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    normalized_arr = (arr - mean_val) / std_val
    return normalized_arr



def moving_average_smooth(arr, window_size):
    window = np.ones(window_size) / window_size
    smoothed_arr = np.convolve(arr, window, mode='same')
    return smoothed_arr


def normalzie(np_l,n_models,samples):

    print("before smooth")
    max_value = np.max(np_l)
    min_value = np.min(np_l)
    print('min_value:',min_value,'max_value:',max_value)

    for l in range(2):
        for i in range(n_models*samples):
            #print(i)
            for j in range(6):
                var = np_l[i][j]
                var = moving_average_smooth(var,5)
                
                for x in range(5):
                    var[x]=var[5].copy()
                    var[19999-x]=var[19994].copy()

                np_l[i][j] = var.copy()

    print("after smooth")
    max_value = np.max(np_l)
    min_value = np.min(np_l)
    print('min_value:',min_value,'max_value:',max_value)

    for i in range(n_models*samples):
        #print(i)
        for j in range(6):
            var = np_l[i][j].copy()
            var = min_max_normalize(var,min_value,max_value)
            np_l[i][j] = var.copy()


    return np_l





def prepare_training(dataset_l,hif='',n_models=15,samples=10):
    global family_based, min_max


    err = 0
    for i in range(n_models*samples):
        if len(dataset_l[i])!=6:
            print("Error: ", i)
            err = 1
        for j in range(6):
            if len(dataset_l[i][j])!=20000:
                print("Error: ", i,j,len(dataset_l[i][j]))
                err = 1
    
    if err:
        sys.exit()
            

    data_array = np.array(dataset_l,dtype=float)

    '''
    print("array after convert to np")
    print(data_array[160][0][0])
    print(data_array[160][5][0])
    print(data_array[0][0][0])
    print(data_array[0][5][0])
    '''

    model = []
    for i in range(n_models):
        for j in range(samples):
            model.append(str(i))

    model = np.array(model)
    print("model.shape", model.shape)



    if min_max!='':
        data_array = normalzie(data_array,n_models,samples)

    if hif != '':
        print("data_array", data_array.shape )
        plt.figure(figsize=(60, 30))
        plt.title(" three dimensions of the three instances in dataset")
        plt.subplot(3, 3, 1)
        plt.plot(data_array[0][0])
        plt.grid(True)

        plt.subplot(3, 3, 2)
        plt.plot(data_array[0][3])
        plt.grid(True)

        plt.subplot(3, 3, 3)
        plt.plot(data_array[0][5])
        plt.grid(True)

        plt.subplot(3, 3, 4)
        plt.plot(data_array[4*samples][0])
        plt.grid(True)

        plt.subplot(3, 3, 5)
        plt.plot(data_array[4*samples][3])
        plt.grid(True)

        plt.subplot(3, 3, 6)
        plt.plot(data_array[4*samples][5])
        plt.grid(True)

        plt.subplot(3, 3, 7)
        plt.plot(data_array[14*samples][0])
        plt.grid(True)

        plt.subplot(3, 3, 8)
        plt.plot(data_array[14*samples][3])
        plt.grid(True)

        plt.subplot(3, 3, 9)
        plt.plot(data_array[14*samples][5])
        plt.grid(True)


        
        #plt.show()
        plt.savefig('sample.png')
        sys.exit()


    return data_array, model




def train_model_ts(X_train, y_train,hif):
    global family_based,min_max, ml_type

    from sktime.classification.kernel_based import RocketClassifier

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    print('time_str',time_str)

    rocket = RocketClassifier(num_kernels=2000)
    print('Start to train...')
    rocket.fit(X_train, y_train)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    print('time_str',time_str)
    

    # Save the trained model to disk
    filename = f'trained_model_{ml_type}{hif}{family_based}{min_max}.joblib'
    joblib.dump(rocket, filename)

    return rocket




def create_confusion(y_true,y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # convert '1' --> '01'
    y_true_t = []
    for i in range(len(y_true)):
        ci = str(int(y_true[i]) + 1)
        if len(ci)==1:
            y_true_t.append('0'+ci)
        else:
            y_true_t.append(ci)


    classes = np.unique(y_true_t)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add labels to each cell
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.savefig('confusion.png')


def eval_model(X_test, y_test,model, test_type):
    print("Lets evaluate:")
    predictions = model.predict(X_test)
    predictions = np.asarray(predictions).tolist()
    y_test = np.asarray(y_test).tolist()
    print(predictions)
    print(y_test)

        
    correct = 0
    f_correct = 0
    for i in range(len(predictions)):

        if(y_test[i]==predictions[i]):
            correct = correct + 1
            f_correct = f_correct + 1
        else:

            if(predictions[i] in get_family(y_test[i],test_type)):
                f_correct = f_correct + 1
                print("Family is corect but ->")

            print("Expected: ", y_test[i], ' , predicted: ',predictions[i])
           
        
    print('Raw accuracy is: ', correct/len(predictions)*100,'%')
    print('Family accuracy is: ', f_correct/len(predictions)*100,'%')

    create_confusion(y_test,predictions)
    
    


#------------ main
train_en = False
ml_type = 'ts'
hif = '' # show only samples
family_based = ''
min_max = ''
re_load = False
test_type = 'u'
for i in range(len(sys.argv)):
    if sys.argv[i]=='train':
        train_en = True
    elif sys.argv[i]=='hif':
        hif = '_hif'
    elif sys.argv[i]=='family':
        family_based = '_family'
    elif sys.argv[i]=='mm':
        min_max = '_mm'
    elif sys.argv[i]=='reload':
        re_load = True
    elif sys.argv[i]=='known':
        test_type = 'k'

if train_en:
    print('training ...')
    
    # model for pc
    dataset_l = prepare_dataset('/Users/usr/Desktop/fingerprinting/regression_results_cpu2/train',re_load,20,17,'')
 
    X_train, y_train = prepare_training(dataset_l,hif,20,17)

    if ml_type=='ts':
        model = train_model_ts(X_train, y_train,hif)


else:
    print('Loading the model for evaluation ...')


    if test_type == 'u':
        # unkown models
        dataset_l = prepare_dataset('/Users/usr/Desktop/fingerprinting/regression_results_cpu2/test/unknown',re_load,11,20,'_utest')
        X_test, y_test = prepare_training(dataset_l,hif,11,20)
    else:
        # known models
        dataset_l = prepare_dataset('/Users/usr/Desktop/fingerprinting/regression_results_cpu2/train',re_load,20,3,'_ktest')
        X_test, y_test = prepare_training(dataset_l,hif,20,3)

    model = joblib.load(f'trained_model_{ml_type}{hif}{family_based}{min_max}.joblib')

    eval_model(X_test, y_test,model, test_type)

