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



#################################
#Plot Country Graphs
#################################

def plot_returns(title, type = 'er_return', set = 'train'):
    %matplotlib qt
    #Find optimal X ticks_to_use
    dates = list(globals()[set + "_AUS"]['ylds_level'].index)
    ticks_location = int(len(dates)/4) - 1
    ticks_to_use = [dates[0], dates[ticks_location], dates[ticks_location*2], dates[ticks_location*3], dates[-1]]

    # Make a data frame
    min_value = 10000000
    max_value = -10000000
    data = {}
    for i in range(len(list_of_countries_considered)):
        data.update({list_of_countries_considered[i]:globals()[set + "_" + list_of_countries_considered[i]][type]})
        if min(globals()[set + "_" + list_of_countries_considered[i]][type]) < min_value:
            min_value = min(globals()[set + "_" + list_of_countries_considered[i]][type])
        if max(globals()[set + "_" + list_of_countries_considered[i]][type]) > max_value:
            max_value = max(globals()[set + "_" + list_of_countries_considered[i]][type])
    df=pd.DataFrame.from_dict(data)

    # Initialize the figure
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('tab20b')

    # multiple line plot
    num=0
    fig = plt.figure(figsize=(20,18))
    for column in df:
        num+=1

        # Find the right spot on the plot
        fig.add_subplot(4,5, num)

        # plot every groups, but discreet
        for v in df:
            plt.plot(df[v], marker='', color='grey', linewidth=0.6, alpha=0.3)


        # Plot the lineplot
        plt.plot(df[column], marker='', color=palette(num), linewidth=2.0, alpha=0.9, label=column)
        plt.locator_params(axis = 'x', nticks=10)

        # Same limits for everybody!
        plt.ylim(min_value,max_value)
        plt.xticks(ticks_to_use)

        # Not ticks everywhere
        if num in range(16) :
            plt.tick_params(labelbottom='off')
        if num not in [1,6,11,16] :
            plt.tick_params(labelleft='off')



        # Add title
        plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )

    # general title
    plt.suptitle(title, fontsize=16, fontweight=0, color='black', style='italic')

    # Axis title
    plt.text(0.5, 0.02, 'Time', ha='center', va='center')
    plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')

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
    globals()["clf_" + list_of_countries_considered[i]] = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
    globals()["clf_" + list_of_countries_considered[i]].fit(train_features, train_target)
    #Organize Results
    results = {"model": globals()["clf_" + list_of_countries_considered[i]],
               "C": globals()["clf_" + list_of_countries_considered[i]].best_estimator_.C,
               "gamma": globals()["clf_" + list_of_countries_considered[i]].best_estimator_.gamma,
               "kernel": globals()["clf_" + list_of_countries_considered[i]].best_estimator_.kernel,
               "train_accuracy": accuracy_score(train_target, globals()["clf_" + list_of_countries_considered[i]].predict(train_features)),
               "test_accuracy": accuracy_score(test_target, globals()["clf_" + list_of_countries_considered[i]].predict(test_features)),
               "train_F1score": f1_score(train_target, globals()["clf_" + list_of_countries_considered[i]].predict(train_features)),
               "test_F1score": f1_score(test_target, globals()["clf_" + list_of_countries_considered[i]].predict(test_features))}
    return results



#############################
#Plot Confusion Matrix
############################

#Plot individual Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title = 'Normalized confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel=f'Predicted label\naccuracy = {round(np.trace(cm) / float(np.sum(cm)), 3)}; missclass = {round(1 - (np.trace(cm) / float(np.sum(cm))), 3)}')

    # Get rid of white grid lines
    plt.grid('off')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#Plot all countries confusion matrices
def plot_confusionMatrices(set = 'test', type = 'clf'):
    %matplotlib qt
    cmap=plt.cm.Blues
    title = ' Normalized Confusion Matrix'
    title = set.capitalize() + title
    axes = []

    #Create a list of axes to be created
    for i in range(4):
        for j in range(5):
            axes.append((i,j))

    fig, axs = plt.subplots(4,5)


    #Create and plot confusion matrices
    if type == 'clf':
        for i in range(len(list_of_countries_considered)):
            globals()["cm_" + list_of_countries_considered[i]] = confusion_matrix(globals()[set + "_" + list_of_countries_considered[i]].iloc[:, -1],
                                  globals()[type + "_" + list_of_countries_considered[i]]['model'].predict(globals()[set + "_" + list_of_countries_considered[i]].iloc[:, 0:9]))
            globals()["cm_" + list_of_countries_considered[i]] = globals()["cm_" + list_of_countries_considered[i]].astype('float') / globals()["cm_" + list_of_countries_considered[i]].sum(axis=1)[:, np.newaxis]
            im = axs[axes[i]].imshow(globals()["cm_" + list_of_countries_considered[i]], interpolation='nearest', cmap=cmap)
            axs[axes[i]].figure.colorbar(im, ax=axs[axes[i]])
    elif type == 'neural':
        for i in range(len(list_of_countries_considered)):
            globals()["cm_" + list_of_countries_considered[i]] = confusion_matrix(globals()[set + "_" + list_of_countries_considered[i]].iloc[:, -1],
                                [1 if prediction[1] > 0.5 else 0 for prediction in globals()[type + "_" + list_of_countries_considered[i]]['model'].predict(globals()[set + "_" + list_of_countries_considered[i]].iloc[:, 0:9])])
            globals()["cm_" + list_of_countries_considered[i]] = globals()["cm_" + list_of_countries_considered[i]].astype('float') / globals()["cm_" + list_of_countries_considered[i]].sum(axis=1)[:, np.newaxis]
            im = axs[axes[i]].imshow(globals()["cm_" + list_of_countries_considered[i]], interpolation='nearest', cmap=cmap)
            axs[axes[i]].figure.colorbar(im, ax=axs[axes[i]])


    #Put values into confusion matrices
    for h in range(len(list_of_countries_considered)):
        fmt = '.2f'
        thresh = globals()["cm_" + list_of_countries_considered[h]].max() / 2.
        for i in range(globals()["cm_" + list_of_countries_considered[h]].shape[0]):
            for j in range(globals()["cm_" + list_of_countries_considered[h]].shape[1]):
                axs[axes[h]].text(j, i, format(globals()["cm_" + list_of_countries_considered[h]][i, j], fmt),
                        ha="center", va="center",
                        color="white" if globals()["cm_" + list_of_countries_considered[h]][i, j] > thresh else "black")

    #Label confusion matrices
    for i in range(len(list_of_countries_considered)):
        axs[axes[i]].set(xticks=np.arange(globals()["cm_" + list_of_countries_considered[i]].shape[1]),
               yticks=np.arange(globals()["cm_" + list_of_countries_considered[i]].shape[0]),
               # ... and label them with the respective list entries
               xticklabels=['1','0'], yticklabels=['1','0'],
               title=list_of_countries_considered[i],
               ylabel='True label',
               xlabel=f'accuracy = {round(np.trace(globals()["cm_" + list_of_countries_considered[i]]) / float(np.sum(globals()["cm_" + list_of_countries_considered[i]])), 3)}')
        axs[axes[i]].grid(False)

    plt.suptitle(title, fontsize=16, fontweight=0, color='black', style='italic')
    fig.tight_layout()

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

    tag_classifier.fit(train_features, train_target, epochs = 500,
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
