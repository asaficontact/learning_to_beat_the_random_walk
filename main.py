from imports import *
from functions_kit import *
from carryTrader import ct
%matplotlib qt

###########################
#Carry Trade
##########################

trade = ct(10000, 1)
x, y , z = trade.basic_trade([50,50])


z["average_percentage_profit"]
z['average_profit']

z['short_country_name_list']

z['long_country_name_list']


%matplotlib qt
plt.plot(y)
plt.axhline(y=0, color='r', linestyle='-')
plt.ylabel("Percentage")
plt.xlabel("Date")
plt.savefig('images/carryTrade/halfYear1.png')


z["average_percentage_profit"]
z['average_profit']









###########################################
#FOREX TRADING
##########################################

##############
#Load Files
##############

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

#Get Exchange Rates from Fred
api = '3dcbccc26181a5457fb8bd0584de00a8'
exchange_rates = get_exchange_rates(api)
exchange_rates.tail()

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

ylds_CZR.head()



# test_ylds_US = ylds_US.iloc[training_length:]
# test_exp_US = exp_US.iloc[training_length:]
# test_tp_US = tp_US.iloc[training_length:]

#Exchange Rate
train_exchange_rates = exchange_rates.iloc[:training_length + 1]
test_exchange_rates = exchange_rates.iloc[training_length:]

#############################
#Merge Datasets
#############################
for i in range(len(list_of_countries_considered)):
    globals()["train_" + list_of_countries_considered[i]] =merge_datasets(globals()['train_ylds_' + list_of_countries_considered[i]],
                                                                globals()['train_exp_' + list_of_countries_considered[i]],
                                                                globals()['train_tp_' + list_of_countries_considered[i]], True,
                                                                train_exchange_rates[list_of_countries_considered[i]])

    globals()["test_" + list_of_countries_considered[i]] =merge_datasets(globals()['test_ylds_' + list_of_countries_considered[i]],
                                                                globals()['test_exp_' + list_of_countries_considered[i]],
                                                                globals()['test_tp_' + list_of_countries_considered[i]], True,
                                                                test_exchange_rates[list_of_countries_considered[i]])


test_AUS.tail()


train_AUS.tail()

#US Dataset
train_US = merge_datasets(train_ylds_US, train_exp_US, train_tp_US, False)
test_US = merge_datasets(test_ylds_US, test_exp_US, test_tp_US, False)

#################################
#Caculate return on exchange rates
#################################
#Formulae = (m[t] - m[t+1])/m[t+1]
for i in range(len(list_of_countries_considered)):
    globals()["train_" + list_of_countries_considered[i]] = calculate_returns(globals()["train_" + list_of_countries_considered[i]],
                                                                                list_of_countries_considered[i], nlag= 1)

for i in range(len(list_of_countries_considered)):
    globals()["test_" + list_of_countries_considered[i]] = calculate_returns(globals()["test_" + list_of_countries_considered[i]],
                                                                                list_of_countries_considered[i], nlag= 1)
train_AUS.tail()

test_AUS.tail()
#Drop US training dataset rows to match that of other Datasets
nlag = 1
test_US = test_US[:-1*nlag]
train_US = train_US[:-1*nlag]

train_US.head()

train_US.tail()

test_US.head()

test_US.tail()
#####################################################
#Calcuate YLDS, EXP, and TP Differential with US
####################################################

for i in range(len(list_of_countries_considered)):
    globals()["train_" + list_of_countries_considered[i]] = calculate_differential(globals()["train_" + list_of_countries_considered[i]], train_US)
    globals()["test_" + list_of_countries_considered[i]] = calculate_differential(globals()["test_" + list_of_countries_considered[i]], test_US)

#################################
#Plot Country Graphs
#################################

def plot_returns(title, type = 'er_return', set = 'train'):
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
        fig.add_subplot(5,4, num)

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
        if num in range(17) :
            plt.tick_params(labelbottom='off')
        if num not in [1,5,9,13,17] :
            plt.tick_params(labelleft='off')



        # Add title
        plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )

    # general title
    plt.suptitle(title, fontsize=16, fontweight=0, color='black', style='italic')

    # Axis title
    plt.text(0.5, 0.02, 'Time', ha='center', va='center')
    plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')
    plt.savefig('images/direction/yield_curvature.png')


plot_returns('', 'ylds_curvature')

%matplotlib qt


train_CAN.head()

#########################################
#Create Buy/Sell column
#########################################
train_buy_distributions = {}
test_buy_distributions = {}

for i in range(len(list_of_countries_considered)):
    globals()["train_" + list_of_countries_considered[i]], globals()["train_buy_" + list_of_countries_considered[i]]= buy_classifier_setup(globals()["train_" + list_of_countries_considered[i]])
    globals()["test_" + list_of_countries_considered[i]], globals()["test_buy_" + list_of_countries_considered[i]] = buy_classifier_setup(globals()["test_" + list_of_countries_considered[i]])
    train_buy_distributions.update({list_of_countries_considered[i]: globals()["train_buy_" + list_of_countries_considered[i]]})
    test_buy_distributions.update({list_of_countries_considered[i]: globals()["test_buy_" + list_of_countries_considered[i]]})

train_buy_distributions
test_buy_distributions

train_AUS.head()

train_US.head()
#############################################################
#Support Vector Classifier with Grid Search for all countries
############################################################

#Fit Support Vectors for all countries:

for i in range(len(list_of_countries_considered)):
    globals()["clf_" + list_of_countries_considered[i]] = support_vectorClassifier(globals()["train_" + list_of_countries_considered[i]],
                                                                                   globals()["test_" + list_of_countries_considered[i]])
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
                    color="orange" if i ==1 and j ==2 else "orange")
    fig.tight_layout()
    plt.savefig('images/mex_neural.png')
    return ax

#
test_CZR.head()

clf_CZR.keys()
neural_MEX.keys()
plot_confusion_matrix(test_MEX['buy'], x1, ["1","0"])

x = neural_MEX['model'].predict(test_MEX.iloc[:, 0:9])
x = np.where(x > 0.5, 1, 0)
x1 = [y[0] for y in x]
x1

train_MEX['buy']




#Plot all countries confusion matrices
def plot_confusionMatrices(set = 'test', type = 'clf'):
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

plot_confusionMatrices('train', type = 'clf')

##################################################
#Create Neural Network Classifier
##################################################
for i in range(len(list_of_countries_considered)):
    globals()["neural_" + list_of_countries_considered[i]] = neural_networkClassifier(globals()["train_" + list_of_countries_considered[i]],
                                                                                   globals()["test_" + list_of_countries_considered[i]])

neural_CAN

plot_confusionMatrices('test', type = 'neural')
