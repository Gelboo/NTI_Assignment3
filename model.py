import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
class My_model:
    def __init__(self):
        pass
    def load_data(self):
        self.data = pd.read_csv('Salaries.csv')
    def show_statistics(self):
        print(self.data.describe())
    def visualize_statistics(self):
        # self.data['phd'].hist()

        x = self.data.index.values
        y = self.data.salary.values
        plt.scatter(x,y,marker='x',c='r')
        plt.xlabel('ID')
        plt.ylabel('Salary')
        plt.title('Academic staff Salary ')
        plt.show()
    def show_data(self):
        print(self.data.head())
    def preprocessing(self):
        # get the string(object) columns in the dataframe
        strings_df_names = list(self.data.select_dtypes(include='object').columns.values)
        # print(strings_df_names)
        self.encoder = {}
        for str_col_name in strings_df_names:
            self.encoder[str_col_name] = LabelEncoder()
            self.data[str_col_name] = self.encoder[str_col_name].fit_transform(self.data[str_col_name])
    def split_data(self):
        from sklearn.model_selection import train_test_split
        x = self.data.iloc[:,:-1].values
        y = self.data.iloc[:,-1].values
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.2)
        # print(self.y_train)
    def train(self,model_name):
        if model_name == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
        elif model_name == "SVR":
            from sklearn.svm import SVR
            self.model = SVR()
        elif model_name == "DecisionTree":
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor()
        elif model_name == "Knn":
            from sklearn.neighbors import KNeighborsRegressor
            self.model = KNeighborsRegressor()
        self.model.fit(self.x_train,self.y_train)
    def compute_accuracy(self):
        from sklearn.metrics import mean_squared_error,r2_score,roc_curve,auc
        y_pred = self.model.predict(self.x_test)
        print(y_pred)
        print(self.y_test)
        MSE = mean_squared_error(y_pred,self.y_test)
        R2 = r2_score(self.y_test,y_pred)
        fpr,tpr,thresholds = roc_curve(self.y_test,y_pred,pos_label=2)
        print("AUC = ",auc(fpr,tpr))
        return MSE,R2
    def predict(self,rank,discipline,phd,service,sex):
        # preprocessing these values
        rank = self.encoder['rank'].transform([rank])[0]
        discipline = self.encoder['discipline'].transform([discipline])[0]
        sex = self.encoder['sex'].transform([sex])[0]
        phd = int(phd)
        service = int(service)
        return self.model.predict(np.array([rank,discipline,phd,service,sex]).reshape(1,-1))
    def serialized(self):
        pickle_save = open("model.pickle","wb")
        pickle.dump(self.model,pickle_save)
        pickle_save.close()
if __name__=='__main__':
    m = My_model()
    m.load_data()
    # m.visualize_statistics()
    print('\n\n')
    m.show_statistics()
    print('\n\n')
    m.show_data()
    print('\n\n')
    m.preprocessing()
    m.show_data()

    m.split_data()
    m.train('Knn')
    m.compute_accuracy()
    m.predict(rank='Prof',discipline='B',phd='56',service='49',sex='Male')
    m.serialized()
