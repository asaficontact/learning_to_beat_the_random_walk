from imports import *
from functions_kit import *

class ft:
    def __init__(self, load = True):
        self.list_of_countries_considered = self.prepareDataset(load)


    def buy_classifier_setup(self, df):
        data = df.copy()
        buy = [1 if exchange_return > 0 else 0 for exchange_return in data['er_return']]
        data['buy'] = buy
        data_distribution = {"one": round(len(data[data['buy'] == 1])/len(data),2),
                            "zero": round(len(data[data['buy'] == 0])/len(data),2)}
        return data, data_distribution

    def prepareDataset(self, load):

        list_of_countries = ["AUS", "CAN", "CHI", "CHN", "COL", "CZR", "EUR", "HUN",
                            "INDO", "JAP", "MAL", "MEX", "NOR", "NZ", "PER", "PHI", "PO",
                            "SA", "SNG","SWE", "SWI", "UK", "US", "BRA"]

        #Read All Yields
        ylds_start_dates = {}
        ylds_end_dates = {}
        xlsx = pd.ExcelFile('data/ALL_YLDS.xlsx')
        for i in range(len(list_of_countries)):
            globals()['ylds_' + list_of_countries[i]] = pd.read_excel(xlsx, list_of_countries[i])
            globals()['ylds_' + list_of_countries[i]] = globals()['ylds_' + list_of_countries[i]].set_index('dates')
            globals()['ylds_' + list_of_countries[i]] = globals()['ylds_' + list_of_countries[i]].iloc[:,1:]
            ylds_start_dates.update({list_of_countries[i]: globals()['ylds_' + list_of_countries[i]].index[0]})
            ylds_end_dates.update({list_of_countries[i]: globals()['ylds_' + list_of_countries[i]].index[-1]})


        #Read All Expectations
        exp_start_dates = {}
        exp_end_dates = {}
        xlsx = pd.ExcelFile('data/ALL_EXP.xlsx')
        for i in range(len(list_of_countries)):
            globals()['exp_' + list_of_countries[i]] = pd.read_excel(xlsx, list_of_countries[i])
            globals()['exp_' + list_of_countries[i]] = globals()['exp_' + list_of_countries[i]].set_index('dates')
            globals()['exp_' + list_of_countries[i]] = globals()['exp_' + list_of_countries[i]].iloc[:,1:]
            exp_start_dates.update({list_of_countries[i]: globals()['exp_' + list_of_countries[i]].index[0]})
            exp_end_dates.update({list_of_countries[i]: globals()['exp_' + list_of_countries[i]].index[-1]})

        #Read All Term Premium
        tp_start_dates = {}
        tp_end_dates = {}
        xlsx = pd.ExcelFile('data/ALL_TP.xlsx')
        for i in range(len(list_of_countries)):
            globals()['tp_' + list_of_countries[i]] = pd.read_excel(xlsx, list_of_countries[i])
            globals()['tp_' + list_of_countries[i]] = globals()['tp_' + list_of_countries[i]].set_index('dates')
            globals()['tp_' + list_of_countries[i]] = globals()['tp_' + list_of_countries[i]].iloc[:,1:]
            tp_start_dates.update({list_of_countries[i]: globals()['tp_' + list_of_countries[i]].index[0]})
            tp_end_dates.update({list_of_countries[i]: globals()['tp_' + list_of_countries[i]].index[-1]})

        if load:
            exchange_rates = pd.read_pickle('data/exchange_rates.pkl')

        else:
            api = '3dcbccc26181a5457fb8bd0584de00a8'
            exchange_rates = get_exchange_rates(api)
            exchange_rates.to_pickle('data/exchange_rates.pkl')

        list_of_countries_considered = ["AUS", "CAN", "CHI", "CHN", "COL", "CZR", "EUR", "HUN",
                            "INDO", "JAP", "MEX", "NOR", "NZ", "PO","SA", "SNG","SWE", "SWI", "UK" ,"BRA"]

        #Shorten Yields, Expectations, and Term Premium as per Brazil Start date:
        start_date = ylds_BRA.index[0]
        end_date = ylds_CHI.index[-1]
        for i in range(len(list_of_countries)):
            globals()['ylds_' + list_of_countries[i]] = globals()['ylds_' + list_of_countries[i]][start_date:end_date]
            globals()['ylds_' + list_of_countries[i]].index = globals()['ylds_' + list_of_countries[i]].index + pd.offsets.MonthBegin(1)

            globals()['tp_' + list_of_countries[i]] = globals()['tp_' + list_of_countries[i]][start_date:end_date]
            globals()['tp_' + list_of_countries[i]].index = globals()['tp_' + list_of_countries[i]].index + pd.offsets.MonthBegin(1)

            globals()['exp_' + list_of_countries[i]] = globals()['exp_' + list_of_countries[i]][start_date:end_date]
            globals()['exp_' + list_of_countries[i]].index = globals()['exp_' + list_of_countries[i]].index + pd.offsets.MonthBegin(1)

        start_date = ylds_BRA.index[0]
        end_date = ylds_CHI.index[-1]
        exchange_rates = exchange_rates[start_date:end_date]

        ############################
        #Principal Component Analysis
        #Along with Train/Test Split
        ############################
        train_data = 0.7
        test_data = 0.3
        training_length = int(train_data * len(ylds_BRA))
        testing_length = int(len(ylds_BRA) - training_length)

        for i in range(len(list_of_countries)):
            globals()['train_ylds_' + list_of_countries[i]] = PCA_analysis(globals()['ylds_' + list_of_countries[i]].iloc[:training_length + 1], 'ylds', False)
            globals()['train_exp_' + list_of_countries[i]] = PCA_analysis(globals()['exp_' + list_of_countries[i]].iloc[:training_length + 1], 'exp', False)
            globals()['train_tp_' + list_of_countries[i]] = PCA_analysis(globals()['tp_' + list_of_countries[i]].iloc[:training_length + 1], 'tp', False)

            globals()['test_ylds_' + list_of_countries[i]] = PCA_analysis(globals()['ylds_' + list_of_countries[i]].iloc[training_length:], 'ylds', False)
            globals()['test_exp_' + list_of_countries[i]] = PCA_analysis(globals()['exp_' + list_of_countries[i]].iloc[training_length:], 'exp', False)
            globals()['test_tp_' + list_of_countries[i]] = PCA_analysis(globals()['tp_' + list_of_countries[i]].iloc[training_length:], 'tp', False)

        #US Dataset
        train_ylds_US = PCA_analysis(ylds_US.iloc[:training_length + 1], 'ylds', False)
        train_exp_US = PCA_analysis(exp_US.iloc[:training_length + 1], 'exp', False)
        train_tp_US = PCA_analysis(tp_US.iloc[:training_length + 1], 'tp', False)

        test_ylds_US = PCA_analysis(ylds_US.iloc[training_length:], 'ylds', False)
        test_exp_US = PCA_analysis(exp_US.iloc[training_length:], 'exp', False)
        test_tp_US = PCA_analysis(tp_US.iloc[training_length:], 'tp', False)


        #Exchange Rate
        train_exchange_rates = exchange_rates.iloc[:training_length + 1]
        test_exchange_rates = exchange_rates.iloc[training_length:]


        #Merge Datasets
        for i in range(len(list_of_countries_considered)):
            globals()["train_" + list_of_countries_considered[i]] =merge_datasets(globals()['train_ylds_' + list_of_countries_considered[i]],
                                                                        globals()['train_exp_' + list_of_countries_considered[i]],
                                                                        globals()['train_tp_' + list_of_countries_considered[i]], True,
                                                                        train_exchange_rates[list_of_countries_considered[i]])

            globals()["test_" + list_of_countries_considered[i]] =merge_datasets(globals()['test_ylds_' + list_of_countries_considered[i]],
                                                                        globals()['test_exp_' + list_of_countries_considered[i]],
                                                                        globals()['test_tp_' + list_of_countries_considered[i]], True,
                                                                        test_exchange_rates[list_of_countries_considered[i]])

        #US Dataset
        train_US = merge_datasets(train_ylds_US, train_exp_US, train_tp_US, False)
        test_US = merge_datasets(test_ylds_US, test_exp_US, test_tp_US, False)


        # Caculate return on exchange rates
        # Formulae = (m[t] - m[t+1])/m[t+1]

        for i in range(len(list_of_countries_considered)):
            globals()["train_" + list_of_countries_considered[i]] = calculate_returns(globals()["train_" + list_of_countries_considered[i]],
                                                                                        list_of_countries_considered[i], nlag= 1)

        for i in range(len(list_of_countries_considered)):
            globals()["test_" + list_of_countries_considered[i]] = calculate_returns(globals()["test_" + list_of_countries_considered[i]],
                                                                                        list_of_countries_considered[i], nlag= 1)

        #Drop US training dataset rows to match that of other Datasets
        nlag = 1
        test_US = test_US[:-1*nlag]
        train_US = train_US[:-1*nlag]

        #####################################################
        #Calcuate YLDS, EXP, and TP Differential with US
        ####################################################

        for i in range(len(list_of_countries_considered)):
            globals()["train_" + list_of_countries_considered[i]] = calculate_differential(globals()["train_" + list_of_countries_considered[i]], train_US)
            globals()["test_" + list_of_countries_considered[i]] = calculate_differential(globals()["test_" + list_of_countries_considered[i]], test_US)


        #########################################
        #Create Buy/Sell column
        #########################################
        for i in range(len(list_of_countries_considered)):
            globals()["train_" + list_of_countries_considered[i]], globals()["train_buy_" + list_of_countries_considered[i]]= self.buy_classifier_setup(globals()["train_" + list_of_countries_considered[i]])
            globals()["test_" + list_of_countries_considered[i]], globals()["test_buy_" + list_of_countries_considered[i]] = self.buy_classifier_setup(globals()["test_" + list_of_countries_considered[i]])

        return list_of_countries_considered




    def plot_returns(self, title, type = 'er_return', set = 'train'):
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




    def support_vectorClassifier(self):
        for i in range(len(self.list_of_countries_considered)):
            #Split dataset
            train_features = globals()["train_" + self.list_of_countries_considered[i]].iloc[:, 0:9]
            train_target = globals()["train_" + self.list_of_countries_considered[i]].iloc[:, -1]
            test_features = globals()["test_" + self.list_of_countries_considered[i]].iloc[:, 0:9]
            test_target = globals()["test_" + self.list_of_countries_considered[i]].iloc[:, -1]
            #Create parameters
            parameter_candidates = [
              {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.00001,0.0001, 0.001, 0.01, 0.1, 1], 'kernel': ['rbf']},
            ]
            #Fit Data
            globals()["clf_" + self.list_of_countries_considered[i]] = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
            globals()["clf_" + self.list_of_countries_considered[i]].fit(train_features, train_target)
            #Organize Results
            globals()["clf_" + self.list_of_countries_considered[i]] = {"model": globals()["clf_" + self.list_of_countries_considered[i]],
                       "C": globals()["clf_" + self.list_of_countries_considered[i]].best_estimator_.C,
                       "gamma": globals()["clf_" + self.list_of_countries_considered[i]].best_estimator_.gamma,
                       "kernel": globals()["clf_" + self.list_of_countries_considered[i]].best_estimator_.kernel,
                       "train_accuracy": accuracy_score(train_target, globals()["clf_" + self.list_of_countries_considered[i]].predict(train_features)),
                       "test_accuracy": accuracy_score(test_target, globals()["clf_" + self.list_of_countries_considered[i]].predict(test_features)),
                       "train_F1score": f1_score(train_target, globals()["clf_" + self.list_of_countries_considered[i]].predict(train_features)),
                       "test_F1score": f1_score(test_target, globals()["clf_" + self.list_of_countries_considered[i]].predict(test_features))}



     #Plot all countries confusion matrices
    def plot_confusionMatrices(self, set = 'test', type = 'clf'):

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
             for i in range(len(self.list_of_countries_considered)):
                 globals()["cm_" + self.list_of_countries_considered[i]] = confusion_matrix(globals()[set + "_" + self.list_of_countries_considered[i]].iloc[:, -1],
                                       globals()[type + "_" + self.list_of_countries_considered[i]]['model'].predict(globals()[set + "_" + self.list_of_countries_considered[i]].iloc[:, 0:9]))
                 globals()["cm_" + self.list_of_countries_considered[i]] = globals()["cm_" + self.list_of_countries_considered[i]].astype('float') / globals()["cm_" + self.list_of_countries_considered[i]].sum(axis=1)[:, np.newaxis]
                 im = axs[axes[i]].imshow(globals()["cm_" + self.list_of_countries_considered[i]], interpolation='nearest', cmap=cmap)
                 axs[axes[i]].figure.colorbar(im, ax=axs[axes[i]])
         elif type == 'neural':
             for i in range(len(self.list_of_countries_considered)):
                 globals()["cm_" + self.list_of_countries_considered[i]] = confusion_matrix(globals()[set + "_" + self.list_of_countries_considered[i]].iloc[:, -1],
                                     [1 if prediction[1] > 0.5 else 0 for prediction in globals()[type + "_" + self.list_of_countries_considered[i]]['model'].predict(globals()[set + "_" + self.list_of_countries_considered[i]].iloc[:, 0:9])])
                 globals()["cm_" + self.list_of_countries_considered[i]] = globals()["cm_" + self.list_of_countries_considered[i]].astype('float') / globals()["cm_" + self.list_of_countries_considered[i]].sum(axis=1)[:, np.newaxis]
                 im = axs[axes[i]].imshow(globals()["cm_" + self.list_of_countries_considered[i]], interpolation='nearest', cmap=cmap)
                 axs[axes[i]].figure.colorbar(im, ax=axs[axes[i]])


         #Put values into confusion matrices
         for h in range(len(self.list_of_countries_considered)):
             fmt = '.2f'
             thresh = globals()["cm_" + self.list_of_countries_considered[h]].max() / 2.
             for i in range(globals()["cm_" + self.list_of_countries_considered[h]].shape[0]):
                 for j in range(globals()["cm_" + self.list_of_countries_considered[h]].shape[1]):
                     axs[axes[h]].text(j, i, format(globals()["cm_" + self.list_of_countries_considered[h]][i, j], fmt),
                             ha="center", va="center",
                             color="white" if globals()["cm_" + self.list_of_countries_considered[h]][i, j] > thresh else "black")

         #Label confusion matrices
         for i in range(len(self.list_of_countries_considered)):
             axs[axes[i]].set(xticks=np.arange(globals()["cm_" + self.list_of_countries_considered[i]].shape[1]),
                    yticks=np.arange(globals()["cm_" + self.list_of_countries_considered[i]].shape[0]),
                    # ... and label them with the respective list entries
                    xticklabels=['1','0'], yticklabels=['1','0'],
                    title=self.list_of_countries_considered[i],
                    ylabel='True label',
                    xlabel=f'accuracy = {round(np.trace(globals()["cm_" + self.list_of_countries_considered[i]]) / float(np.sum(globals()["cm_" + self.list_of_countries_considered[i]])), 3)}')
             axs[axes[i]].grid(False)

         plt.suptitle(title, fontsize=16, fontweight=0, color='black', style='italic')
         fig.tight_layout()


    def neural_networkClassifier(self):

        for i in range(len(self.list_of_countries_considered)):
            #Split dataset
            train_features = globals()["train_" + self.list_of_countries_considered[i]].iloc[:, 0:9]
            train_target = globals()["train_" + self.list_of_countries_considered[i]].iloc[:, -1]
            test_features = globals()["test_" + self.list_of_countries_considered[i]].iloc[:, 0:9]
            test_target = globals()["test_" + self.list_of_countries_considered[i]].iloc[:, -1]

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
            globals()["neural_" + self.list_of_countries_considered[i]] = {"model": tag_classifier,
                       "train_accuracy": accuracy_score(train_df.iloc[:, -1], train_y_pred),
                       "test_accuracy": accuracy_score(test_df.iloc[:, -1], test_y_pred),
                       "train_F1score": f1_score(train_df.iloc[:, -1], train_y_pred),
                       "test_F1score": f1_score(test_df.iloc[:, -1], test_y_pred)}
