class kBest():
    def rfeFeature(indep_X,dep_y,n):
        rfe_list=[]
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC        
        from sklearn.tree import DecisionTreeClassifier 
        from sklearn.ensemble import RandomForestClassifier        
        from sklearn.feature_selection import chi2, RFE
        
        log_model = LogisticRegression(solver='lbfgs')
        RF_model = RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=0)                               
        DT_model=DecisionTreeClassifier(criterion='gini', max_features='sqrt', splitter='best', random_state=0)
        SVC_model = SVC(kernel='linear', random_state=0)

        model_list = [log_model,RF_model,DT_model, SVC_model]
        for i in model_list:
            log_rfe = RFE(estimator=i, n_features_to_select=n)
            log_fit = log_rfe.fit(indep_X, dep_y)
            log_rfe_feature=log_fit.transform(indep_X)
            rfe_list.append(log_rfe_feature)
        return rfe_list
        
    def selectkbest(indep_X,dep_y,n):
        from sklearn.feature_selection import SelectKBest, chi2, RFE
        test = SelectKBest(score_func=chi2, k=n)
        fit1= test.fit(indep_X,dep_Y)
        selectk_features = fit1.transform(indep_X)
        return selectk_features

    def split_scalar(indep_X,dep_y):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_y, test_size = 0.25, random_state = 0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)    
        return X_train, X_test, y_train, y_test

    def cm_prediction(classifier,X_test, y_test):
        y_pred = classifier.predict(X_test)
            
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
            
        from sklearn.metrics import accuracy_score 
        from sklearn.metrics import classification_report 
            
        Accuracy=accuracy_score(y_test, y_pred )
            
        report=classification_report(y_test, y_pred)
        return  classifier,Accuracy,report,X_test,y_test,cm

    def logistic(X_train,y_train,X_test,y_test):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=kBest.cm_prediction(classifier,X_test, y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm      

    def svm_linear(X_train,y_train,X_test,y_test):
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=kBest.cm_prediction(classifier,X_test, y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm

    def svm_NL(X_train,y_train,X_test,y_test):
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=kBest.cm_prediction(classifier,X_test, y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm

    def Navie(X_train,y_train,X_test,y_test):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=kBest.cm_prediction(classifier,X_test, y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm

    def knn(X_train,y_train,X_test,y_test):
        # Fitting K-NN to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=kBest.cm_prediction(classifier,X_test, y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm

    def Decision(X_train,y_train,X_test,y_test):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=kBest.cm_prediction(classifier,X_test, y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm    

    def random(X_train,y_train,X_test,y_test):
        # Fitting K-NN to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=kBest.cm_prediction(classifier,X_test, y_test)
        return  classifier,Accuracy,report,X_test,y_test,cm

    def selectk_Classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf): 
        import pandas as pd
        dataframe=pd.DataFrame(index=['ChiSquare'],columns=['Logistic','SVMl','SVMnl','KNN','Navie','Decision','Random'])
        for number,idex in enumerate(dataframe.index):      
            dataframe['Logistic'][idex]=acclog[number]       
            dataframe['SVMl'][idex]=accsvml[number]
            dataframe['SVMnl'][idex]=accsvmnl[number]
            dataframe['KNN'][idex]=accknn[number]
            dataframe['Navie'][idex]=accnav[number]
            dataframe['Decision'][idex]=accdes[number]
            dataframe['Random'][idex]=accrf[number]
        return dataframe

    def rfe_Classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf): 
        import pandas as pd
        rfe_dataframe=pd.DataFrame(index=['Logistic','SVC','DecisionTree','RandomForest'],columns=['Logistic','SVMl','SVMnl','KNN','Navie','Decision', 'Random'])
        for number,idex in enumerate(rfe_dataframe.index):      
            rfe_dataframe['Logistic'][idex]=acclog[number]       
            rfe_dataframe['SVMl'][idex]=accsvml[number]
            rfe_dataframe['SVMnl'][idex]=accsvmnl[number]
            rfe_dataframe['KNN'][idex]=accknn[number]
            rfe_dataframe['Navie'][idex]=accnav[number]
            rfe_dataframe['Decision'][idex]=accdes[number]
            rfe_dataframe['Random'][idex]=accrf[number]
        return rfe_dataframe
