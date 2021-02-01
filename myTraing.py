import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == '__main__':
    df=pd.read_csv('Data.csv')
    train ,test= data_split(df,0.2)
    X_train=train[['fever','bodypain','age','runnyNose','diffBreath']].to_numpy()
    X_test=test[['fever','bodypain','age','runnyNose','diffBreath']].to_numpy()


    y_train=train[['infectionProb']].to_numpy().reshape(2060,)
    y_test=test[['infectionProb']].to_numpy().reshape(515,) 


    

    clf=LogisticRegression()
    clf.fit(X_train,y_train)


    file=open('model.pkl','wb')
    pickle.dump(clf,file)
    file.close()


