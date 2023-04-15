import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class knn():
    distances = [[]*2]
    final_label = [] 


    def euclidean_distance(self,test_vector,train_feature_vectors,train_labels):
        for i in range(len(train_feature_vectors)):
            distance= np.sqrt(np.sum((test_vector-train_feature_vectors[i])**2))
            self.distances.append([distance,train_labels[i]])
        self.distances = sorted(self.distances, key=lambda x:x[0])
      

    def get_k_nearest_neighbours(self,k):
        labels = []
        for i in range(k):
            labels.append(self.distances[i][1])#get only classes
        return labels

    def getNearestNeighbor(self,k,Trainlabels):
        labels = self.get_k_nearest_neighbours(k)
        freq = [0]*2 #holds [number of class 0, number of class 1] in the nearest k neighbours
            
        for i in range(len(labels)):
            if labels[i] == 0:
                freq[0]+=1
            else:
                freq[1]+=1

        if freq[0]==freq[1]:
            return Trainlabels[0]
        return max(set(labels), key=labels.count)


    def Classifier(self,k,train_features, test_features, Trainlabels):
        for test in test_features:
            self.distances = []
            self.euclidean_distance(test,train_features,Trainlabels)
            self.final_label.append(self.getNearestNeighbor(k,Trainlabels))
        return self.final_label
    
    def Normalize(self,feature):
        normalizeArr =[]
        for idx in feature:#loop over the 4 featuers
            temp = (idx - np.average(feature)) / np.std(feature)
            normalizeArr.append(temp)
        return normalizeArr

    def loadD(self ,data):    

            X = np.delete(data, 4, axis=1)#delete the target column
            y = []
            for i in range(len(data)):
                y.append(data[i][4])# append the target column

            train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.3,shuffle=True)
            for i in range(4):#normalize column by column
                train_data[:,[i]] = self.Normalize(train_data[:,[i]])
                test_data[:, [i]] = self.Normalize(test_data[:,[i]])

            return train_data, test_data, train_labels, test_labels



if __name__ == "__main__":

    data = pd.read_csv('BankNote_Authentication.csv')
    
    Model = knn()
    train_data, test_data, train_labels, test_labels = Model.loadD(data.values)
    k = 3
    KNN_Predictions=knn.Classifier(Model,k,train_data, test_data, train_labels)
    
    wrong_classifier =0
    for i in range(len(test_labels)):
        if KNN_Predictions[i] != test_labels[i]:
            wrong_classifier+=1

    print("K = ",k)
    print("Number of correctly classified instances :",len(test_labels) - wrong_classifier)
    print("Total number of instances : ",len(test_labels))
    accuracy = (1-(wrong_classifier/len(test_labels)))*100
    print("Accuracy: ", accuracy , "%")