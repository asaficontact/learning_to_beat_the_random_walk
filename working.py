from imports import *
from functions_kit import *
from carryTrader import ct

###########################
#Carry Trade
##########################

trade = ct(100000, 1)
x, y , z = trade.basic_trade([40,30,30])


z["average_percentage_profit"]
z['average_profit']

z['long_country_name_list']



%matplotlib qt
plt.plot(y)
plt.axhline(y=0, color='r', linestyle='-')
plt.title("Percentage Profit From Monthly Carry Trade")

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

#For Carry Trade
for i in range(len(list_of_countries)):
    globals()['train_ylds_' + list_of_countries[i]]['shortRun_interest'] = globals()['ylds_' + list_of_countries[i]].iloc[:training_length + 1]['V3']
    globals()['test_ylds_' + list_of_countries[i]]['shortRun_interest'] = globals()['ylds_' + list_of_countries[i]].iloc[training_length:]['V3']

#US Dataset
train_ylds_US = PCA_analysis(ylds_US.iloc[:training_length + 1], 'ylds', False)
train_exp_US = PCA_analysis(exp_US.iloc[:training_length + 1], 'exp', False)
train_tp_US = PCA_analysis(tp_US.iloc[:training_length + 1], 'tp', False)

test_ylds_US = PCA_analysis(ylds_US.iloc[training_length:], 'ylds', False)
test_exp_US = PCA_analysis(exp_US.iloc[training_length:], 'exp', False)
test_tp_US = PCA_analysis(tp_US.iloc[training_length:], 'tp', False)

#US for Carry Trade
train_ylds_US['shortRun_interest'] = ylds_US.iloc[:training_length + 1]['V3']
test_ylds_US['shortRun_interest'] = ylds_US.iloc[training_length:]['V3']

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

train_US.tail()

#####################################################
#Calcuate YLDS, EXP, and TP Differential with US
####################################################

for i in range(len(list_of_countries_considered)):
    globals()["train_" + list_of_countries_considered[i]] = calculate_differential(globals()["train_" + list_of_countries_considered[i]], train_US)
    globals()["test_" + list_of_countries_considered[i]] = calculate_differential(globals()["test_" + list_of_countries_considered[i]], test_US)

#################################
#Plot Country Graphs
#################################

plot_returns("One Month Ahead Foreign Exchange Return", 'er_return')


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

plot_confusionMatrices('test')

##################################################
#Create Neural Network Classifier
##################################################
for i in range(len(list_of_countries_considered)):
    globals()["neural_" + list_of_countries_considered[i]] = neural_networkClassifier(globals()["train_" + list_of_countries_considered[i]],
                                                                                   globals()["test_" + list_of_countries_considered[i]])

neural_CAN

plot_confusionMatrices('test', type = 'neural')
