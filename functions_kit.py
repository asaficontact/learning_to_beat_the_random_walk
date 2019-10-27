from imports import *

def get_exchange_rates(api):
    fred = Fred(api_key=api)

    #Australia
    AUS_exchange = fred.get_series('EXUSAL')
    AUS_exchange = pd.DataFrame(AUS_exchange)
    AUS_exchange = AUS_exchange.rename(columns = {0:"AUS"})
    AUS_exchange.AUS = 1/AUS_exchange.AUS

    #Brazil
    BRA_exchange = fred.get_series('EXBZUS')
    BRA_exchange = pd.DataFrame(BRA_exchange)
    BRA_exchange = BRA_exchange.rename(columns = {0:"BRA"})

    #CANADA
    CAN_exchange = fred.get_series('EXCAUS')
    CAN_exchange = pd.DataFrame(CAN_exchange)
    CAN_exchange = CAN_exchange.rename(columns = {0:"CAN"})

    #CHILE
    CHI_exchange = fred.get_series('CCUSSP02CLM650N')
    CHI_exchange = pd.DataFrame(CHI_exchange)
    CHI_exchange = CHI_exchange.rename(columns = {0:"CHI"})

    #CHINA
    CHN_exchange = fred.get_series('EXCHUS')
    CHN_exchange = pd.DataFrame(CHN_exchange)
    CHN_exchange = CHN_exchange.rename(columns = {0:"CHN"})

    #COLOMBIA
    COL_exchange = fred.get_series('COLCCUSMA02STM')
    COL_exchange = pd.DataFrame(COL_exchange)
    COL_exchange = COL_exchange.rename(columns = {0:"COL"})

    #CZECH REPUBLIC
    CZR_exchange = fred.get_series('CCUSMA02CZM618N')
    CZR_exchange = pd.DataFrame(CZR_exchange)
    CZR_exchange = CZR_exchange.rename(columns = {0:"CZR"})

    #EURO
    EUR_exchange = fred.get_series('EXUSEU')
    EUR_exchange = pd.DataFrame(EUR_exchange)
    EUR_exchange = EUR_exchange.rename(columns = {0:"EUR"})
    EUR_exchange.EUR = 1/EUR_exchange.EUR

    #Hungary
    HUN_exchange = fred.get_series('CCUSMA02HUM618N')
    HUN_exchange = pd.DataFrame(HUN_exchange)
    HUN_exchange = HUN_exchange.rename(columns = {0:"HUN"})

    #INDONESIA
    INDO_exchange = fred.get_series('CCUSSP02IDM650N')
    INDO_exchange = pd.DataFrame(INDO_exchange)
    INDO_exchange = INDO_exchange.rename(columns = {0:"INDO"})

    #JAPAN
    JAP_exchange = fred.get_series('EXJPUS')
    JAP_exchange = pd.DataFrame(JAP_exchange)
    JAP_exchange = JAP_exchange.rename(columns = {0:"JAP"})

    #MALAYA
    #No Data for it on Fred

    #MEXICO
    MEX_exchange = fred.get_series('EXMXUS')
    MEX_exchange = pd.DataFrame(MEX_exchange)
    MEX_exchange = MEX_exchange.rename(columns = {0:"MEX"})

    #NORWAY
    NOR_exchange = fred.get_series('EXNOUS')
    NOR_exchange = pd.DataFrame(NOR_exchange)
    NOR_exchange = NOR_exchange.rename(columns = {0:"NOR"})

    #New Zealand
    NZ_exchange = fred.get_series('EXUSNZ')
    NZ_exchange = pd.DataFrame(NZ_exchange)
    NZ_exchange = NZ_exchange.rename(columns = {0:"NZ"})
    NZ_exchange.NZ = 1/NZ_exchange.NZ

    #PERU
    #No Data for it on Fred

    #Philippines
    #No Data for it on Fred

    #POLAND
    PO_exchange = fred.get_series('CCUSMA02PLM618N')
    PO_exchange = pd.DataFrame(PO_exchange)
    PO_exchange = PO_exchange.rename(columns = {0:"PO"})

    #South Africa
    SA_exchange = fred.get_series('EXSFUS')
    SA_exchange = pd.DataFrame(SA_exchange)
    SA_exchange = SA_exchange.rename(columns = {0:"SA"})

    #Singapore
    SNG_exchange = fred.get_series('EXSIUS')
    SNG_exchange = pd.DataFrame(SNG_exchange)
    SNG_exchange = SNG_exchange.rename(columns = {0:"SNG"})

    #SWEDEN
    SWE_exchange = fred.get_series('EXSDUS')
    SWE_exchange = pd.DataFrame(SWE_exchange)
    SWE_exchange = SWE_exchange.rename(columns = {0:"SWE"})

    #SWITZERLAND
    SWI_exchange = fred.get_series('EXSZUS')
    SWI_exchange = pd.DataFrame(SWI_exchange)
    SWI_exchange = SWI_exchange.rename(columns = {0:"SWI"})

    #UNITED KINGDOMS
    UK_exchange = fred.get_series('EXUSUK')
    UK_exchange = pd.DataFrame(UK_exchange)
    UK_exchange = UK_exchange.rename(columns = {0:"UK"})
    UK_exchange.UK = 1/UK_exchange.UK

    data_frames = [AUS_exchange, BRA_exchange, CAN_exchange, CHI_exchange, CHN_exchange,
                   COL_exchange, CZR_exchange, EUR_exchange, HUN_exchange, INDO_exchange, MEX_exchange,
                   NOR_exchange, NZ_exchange, PO_exchange, SA_exchange, SNG_exchange, SWE_exchange,
                   SWI_exchange, UK_exchange, JAP_exchange]


    df_merged = reduce(lambda  left,right: pd.merge(left,right,left_index = True, right_index = True,
                                                how='inner'), data_frames)

    return df_merged


#Standardize Data
def standardize_data(df):
    column_names = list(df.columns)
    x = StandardScaler().fit_transform(df.values)
    result = pd.DataFrame(data = x, columns = column_names)
    result.index = df.index
    return result


#Calcuate PCA
def PCA_analysis(df, type, standardize = True):
    if standardize == True:
        data = standardize_data(df)
    else:
        data = df
    pca = PCA(n_components = 3)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = [type + '_level', type + '_slope', type+ '_curvature'])
    principalDf.index = df.index
    return principalDf


#Merge Datasets
def merge_datasets(ylds_df, exp_df, tp_df, exchange = True, exchange_rates_df = None):
    if exchange == True:
        data_frames = [ylds_df, exp_df, tp_df, exchange_rates_df]
    else:
        data_frames = [ylds_df, exp_df, tp_df]
    result = reduce(lambda  left,right: pd.merge(left,right,left_index = True, right_index = True,
                                                how='outer'), data_frames)
    return result


#Calculate Exchange Rate returns
def calculate_returns(df_input, country_name, nlag = 1):
    df = df_input.copy()
    remaining_data = df.iloc[:,:-1]
    remaining_data.drop(remaining_data.index[(-1*nlag):], inplace = True)
    begin_data = df[country_name][:(-1*nlag)]
    end_data = df[country_name][nlag:]
    dates = list(begin_data.index)
    begin_data.reset_index(drop = True, inplace = True)
    end_data.reset_index(drop = True, inplace = True)
    returns = ((begin_data-end_data)/end_data)*100
    result = {'dates': dates, 'er_return': returns}
    result = pd.DataFrame.from_dict(result)
    result.set_index('dates', inplace = True)
    final_result = pd.merge(remaining_data, result, left_index=True, right_index = True, how = 'inner')
    return final_result


#Calcuate YLDS, EXP, and TP Differential with US
def calculate_differential(df, US_df, type = 'forex'):

    data = df.iloc[:,:-1]
    US_data = US_df.copy()
    result = data.subtract(US_data, axis = 'index')
    if type == 'forex':
        result.columns = ['ylds_level_diff','ylds_slope_diff','ylds_curvature_diff','exp_level_diff',
                                'exp_slope_diff','exp_curvature_diff',
                                'tp_level_diff', 'tp_slope_diff',
                                'tp_curvature_diff']
        final_result = pd.merge(result, df, left_index=True, right_index = True, how = 'outer')
        return final_result
    else:
        result.columns = ['ylds_level_diff','ylds_slope_diff','ylds_curvature_diff','shortRun_interest_diff','exp_level_diff',
                                'exp_slope_diff','exp_curvature_diff',
                                'tp_level_diff', 'tp_slope_diff',
                                'tp_curvature_diff']
        final_result = pd.merge(result, df, left_index=True, right_index = True, how = 'outer')
        return final_result





#########################################
#Create Buy/Sell column
#########################################

def buy_classifier_setup(df):
    data = df.copy()
    buy = [1 if exchange_return > 0 else 0 for exchange_return in data['er_return']]
    data['buy'] = buy
    data_distribution = {"one": round(len(data[data['buy'] == 1])/len(data),2),
                        "zero": round(len(data[data['buy'] == 0])/len(data),2)}
    return data, data_distribution


#############################################################
#Support Vector Classifier with Grid Search for all countries
############################################################

def support_vectorClassifier(train_df, test_df):
    #Split dataset
    train_features = train_df.iloc[:, 0:9]
    train_target = train_df.iloc[:, -1]
    test_features = test_df.iloc[:, 0:9]
    test_target = test_df.iloc[:, -1]
    #Create parameters
    parameter_candidates = [
      {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.00001,0.0001, 0.001, 0.01, 0.1, 1], 'kernel': ['rbf']},
    ]
    #Fit Data
    clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
    clf.fit(train_features, train_target)
    #Organize Results
    results = {"model": clf,
               "C": clf.best_estimator_.C,
               "gamma": clf.best_estimator_.gamma,
               "kernel": clf.best_estimator_.kernel,
               "train_accuracy": accuracy_score(train_target, clf.predict(train_features)),
               "test_accuracy": accuracy_score(test_target, clf.predict(test_features)),
               "train_F1score": f1_score(train_target, clf.predict(train_features)),
               "test_F1score": f1_score(test_target, clf.predict(test_features))}
    return results

##################################################
#Create Neural Network Classifier
##################################################

def neural_networkClassifier(train_df, test_df):
    #Split dataset
    train_features = train_df.iloc[:, 0:9]
    train_target = train_df.iloc[:, -1]
    test_features = test_df.iloc[:, 0:9]
    test_target = test_df.iloc[:, -1]

    #Creating Neural Network
    tag_classifier = Sequential()
    #first layer
    tag_classifier.add(Dense(64, activation='relu', kernel_initializer='random_normal', input_dim=9))
    tag_classifier.add(Dropout(.2))
    #second layer
    tag_classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    tag_classifier.add(Dropout(.2))

    #output layer
    #softmax sums predictions to 1, good for multi-classification
    tag_classifier.add(Dense(2, activation ='sigmoid', kernel_initializer='random_normal'))

    #Compiling
    #adam optimizer adjusts learning rate throughout training
    #loss function categorical crossentroy for classification
    tag_classifier.compile(optimizer ='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
    early_stop = EarlyStopping(monitor = 'loss', patience = 1, verbose = 2)

    train_target = to_categorical(train_target) #First column is for 0 second is for 1

    tag_classifier.fit(train_features, train_target, epochs = 1000,
                      batch_size = 10000, verbose = 2,
                      callbacks = [early_stop])

    train_y_pred=tag_classifier.predict(train_features)
    train_y_pred =[1 if prediction[1] > 0.5 else 0 for prediction in train_y_pred]

    test_y_pred=tag_classifier.predict(test_features)
    test_y_pred =[1 if prediction[1] > 0.5 else 0 for prediction in test_y_pred]


    #Organize Results
    results = {"model": tag_classifier,
               "train_accuracy": accuracy_score(train_df.iloc[:, -1], train_y_pred),
               "test_accuracy": accuracy_score(test_df.iloc[:, -1], test_y_pred),
               "train_F1score": f1_score(train_df.iloc[:, -1], train_y_pred),
               "test_F1score": f1_score(test_df.iloc[:, -1], test_y_pred)}
    return results
