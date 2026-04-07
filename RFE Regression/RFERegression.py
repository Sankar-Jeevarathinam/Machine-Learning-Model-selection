class rfeRegression():
    def split_scalar(indep_X,dep_Y):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.25, random_state = 0)
    
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)    
        return X_train, X_test, y_train, y_test

    def r2_prediction(regressor,X_test,y_test):
        y_pred = regressor.predict(X_test)
        from sklearn.metrics import r2_score
        r2=r2_score(y_test,y_pred)
        return r2

    def Linear(X_train,y_train,X_test,y_test):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        r2=rfeRegression.r2_prediction(regressor,X_test,y_test)
        return  r2 

    def svm_linear(X_train,y_train,X_test,y_test):
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'linear')
        regressor.fit(X_train, y_train)
        r2=rfeRegression.r2_prediction(regressor,X_test,y_test)
        return  r2 

    def svm_NL(X_train,y_train,X_test,y_test):
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train)
        r2=rfeRegression.r2_prediction(regressor,X_test,y_test)
        return  r2

    def Decision(X_train,y_train,X_test,y_test):
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(X_train, y_train)
        r2=rfeRegression.r2_prediction(regressor,X_test,y_test)
        return  r2

    def random(X_train,y_train,X_test,y_test):
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(X_train, y_train)
        r2=rfeRegression.r2_prediction(regressor,X_test,y_test)
        return  r2

    def rfeFeature(indep_X,dep_y,n):
        from sklearn.feature_selection import RFE
        rfelist=[]
            
        from sklearn.linear_model import LinearRegression
        lin = LinearRegression()
            
        from sklearn.svm import SVR
        SVRl = SVR(kernel = 'linear')
        SVRnl = SVR(kernel = 'rbf')
            
        from sklearn.tree import DecisionTreeRegressor
        dec = DecisionTreeRegressor(random_state = 0)
            
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators = 10, random_state = 0)
            
        rfemodellist=[lin,SVRl,dec,rf] 
        for i in   rfemodellist:
            #print(i)
            log_rfe = RFE(estimator=i, n_features_to_select=n)
            log_fit = log_rfe.fit(indep_X, dep_y)
            log_rfe_feature=log_fit.transform(indep_X)
            rfelist.append(log_rfe_feature)
        return rfelist

    def rfe_regression(acclog,accsvml,accdes,accrf): 
        import pandas as pd
        rfedataframe=pd.DataFrame(index=['Linear','SVC','Random','DecisionTree'],columns=['Linear','SVMl',
                                                                                            'Decision','Random'])
    
        for number,idex in enumerate(rfedataframe.index):
            rfedataframe['Linear'][idex]=acclog[number]       
            rfedataframe['SVMl'][idex]=accsvml[number]
            rfedataframe['Decision'][idex]=accdes[number]
            rfedataframe['Random'][idex]=accrf[number]
        return rfedataframe