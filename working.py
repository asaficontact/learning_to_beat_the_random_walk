from imports import *
from functions_kit import *


##############
#Load Files
##############

list_of_countries = ["AUS", "BRA", "CAN", "CHI", "CHN", "COL", "CZR", "EUR", "HUN",
                    "INDO", "JAP", "MAL", "MEX", "NOR", "NZ", "PER", "PHI", "PO",
                    "SA", "SNG","SWE", "SWI", "UK", "US"]

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
api = '94fe11e100cc5f852f471a8b884d1ba9'
exchange_rates = get_exchange_rates(api)

exchange_rates.head()

list_of_countries_considered = ["AUS", "BRA", "CAN", "CHI", "CHN", "COL", "CZR", "EUR", "HUN",
                    "INDO", "JAP", "MEX", "NOR", "NZ", "PO","SA", "SNG","SWE", "SWI", "UK"]

#Shorten Yields, Expectations, and Term Premium as per Brazil Start date:
for i in range(len(list_of_countries)):
    globals()['ylds_' + list_of_countries[i]] = globals()['ylds_' + list_of_countries[i]][ylds_BRA.index[0]:ylds_CHI.index[-1]]
    globals()['ylds_' + list_of_countries[i]].index = globals()['ylds_' + list_of_countries[i]].index + pd.offsets.MonthBegin(0)

    globals()['tp_' + list_of_countries[i]] = globals()['tp_' + list_of_countries[i]][ylds_BRA.index[0]:ylds_CHI.index[-1]]
    globals()['tp_' + list_of_countries[i]].index = globals()['tp_' + list_of_countries[i]].index + pd.offsets.MonthBegin(0)

    globals()['exp_' + list_of_countries[i]] = globals()['exp_' + list_of_countries[i]][ylds_BRA.index[0]:ylds_CHI.index[-1]]
    globals()['exp_' + list_of_countries[i]].index = globals()['exp_' + list_of_countries[i]].index + pd.offsets.MonthBegin(0)

exchange_rates = exchange_rates[ylds_BRA.index[0]:ylds_CHI.index[-1]]
exchange_rates = exchange_rates.iloc[1:,:]
ylds_BRA = ylds_BRA.iloc[1:,:]



############################
#Principal Component Analysis
############################

for i in range(len(list_of_countries)):
    globals()['ylds_' + list_of_countries[i]] = PCA_analysis(globals()['ylds_' + list_of_countries[i]], 'ylds', False)
    globals()['exp_' + list_of_countries[i]] = PCA_analysis(globals()['exp_' + list_of_countries[i]], 'exp', False)
    globals()['tp_' + list_of_countries[i]] = PCA_analysis(globals()['tp_' + list_of_countries[i]], 'tp', False)


#############################
#Merge Datasets
#############################

for i in range(len(list_of_countries_considered)):
    globals()[list_of_countries_considered[i]] =merge_datasets(globals()['ylds_' + list_of_countries_considered[i]],
                                                                globals()['exp_' + list_of_countries_considered[i]],
                                                                globals()['tp_' + list_of_countries_considered[i]],
                                                                exchange_rates[list_of_countries_considered[i]])

AUS.head()


#################################
#Caculate return on exchange rates
#################################
for i in range(len(list_of_countries_considered)):
    globals()[list_of_countries_considered[i]] = calculate_returns(globals()[list_of_countries_considered[i]], list_of_countries_considered[i], nlag= 1)


MEX.head()

#Plot returns
def plot_returns(title, type = 'er_return'):
    %matplotlib qt
    # Make a data frame
    min_value = 10000000
    max_value = -10000000
    if type == 'ylds_level':
        data = {}
        for i in range(len(list_of_countries_considered)):
            data.update({list_of_countries_considered[i]:globals()[list_of_countries_considered[i]]['ylds_level']})
            if min(globals()[list_of_countries_considered[i]]['ylds_level']) < min_value:
                min_value = min(globals()[list_of_countries_considered[i]]['ylds_level'])
            if max(globals()[list_of_countries_considered[i]]['ylds_level']) > max_value:
                max_value = max(globals()[list_of_countries_considered[i]]['ylds_level'])
        df=pd.DataFrame.from_dict(data)

    elif type == 'ylds_slope':
        data = {}
        for i in range(len(list_of_countries_considered)):
            data.update({list_of_countries_considered[i]:globals()[list_of_countries_considered[i]]['ylds_slope']})
            if min(globals()[list_of_countries_considered[i]]['ylds_slope']) < min_value:
                min_value = min(globals()[list_of_countries_considered[i]]['ylds_slope'])
            if max(globals()[list_of_countries_considered[i]]['ylds_slope']) > max_value:
                max_value = max(globals()[list_of_countries_considered[i]]['ylds_slope'])
        df=pd.DataFrame.from_dict(data)

    elif type == 'ylds_curvature':
        data = {}
        for i in range(len(list_of_countries_considered)):
            data.update({list_of_countries_considered[i]:globals()[list_of_countries_considered[i]]['ylds_curvature']})
            if min(globals()[list_of_countries_considered[i]]['ylds_curvature']) < min_value:
                min_value = min(globals()[list_of_countries_considered[i]]['ylds_curvature'])
            if max(globals()[list_of_countries_considered[i]]['ylds_curvature']) > max_value:
                max_value = max(globals()[list_of_countries_considered[i]]['ylds_curvature'])
        df=pd.DataFrame.from_dict(data)

    elif type == 'exp_level':
        data = {}
        for i in range(len(list_of_countries_considered)):
            data.update({list_of_countries_considered[i]:globals()[list_of_countries_considered[i]]['exp_level']})
            if min(globals()[list_of_countries_considered[i]]['exp_level']) < min_value:
                min_value = min(globals()[list_of_countries_considered[i]]['exp_level'])
            if max(globals()[list_of_countries_considered[i]]['exp_level']) > max_value:
                max_value = max(globals()[list_of_countries_considered[i]]['exp_level'])
        df=pd.DataFrame.from_dict(data)

    elif type == 'exp_slope':
        data = {}
        for i in range(len(list_of_countries_considered)):
            data.update({list_of_countries_considered[i]:globals()[list_of_countries_considered[i]]['exp_slope']})
            if min(globals()[list_of_countries_considered[i]]['exp_slope']) < min_value:
                min_value = min(globals()[list_of_countries_considered[i]]['exp_slope'])
            if max(globals()[list_of_countries_considered[i]]['exp_slope']) > max_value:
                max_value = max(globals()[list_of_countries_considered[i]]['exp_slope'])
        df=pd.DataFrame.from_dict(data)

    elif type == 'exp_curvature':
        data = {}
        for i in range(len(list_of_countries_considered)):
            data.update({list_of_countries_considered[i]:globals()[list_of_countries_considered[i]]['exp_curvature']})
            if min(globals()[list_of_countries_considered[i]]['exp_curvature']) < min_value:
                min_value = min(globals()[list_of_countries_considered[i]]['exp_curvature'])
            if max(globals()[list_of_countries_considered[i]]['exp_curvature']) > max_value:
                max_value = max(globals()[list_of_countries_considered[i]]['exp_curvature'])
        df=pd.DataFrame.from_dict(data)

    elif type == 'tp_level':
        data = {}
        for i in range(len(list_of_countries_considered)):
            data.update({list_of_countries_considered[i]:globals()[list_of_countries_considered[i]]['tp_level']})
            if min(globals()[list_of_countries_considered[i]]['tp_level']) < min_value:
                min_value = min(globals()[list_of_countries_considered[i]]['tp_level'])
            if max(globals()[list_of_countries_considered[i]]['tp_level']) > max_value:
                max_value = max(globals()[list_of_countries_considered[i]]['tp_level'])
        df=pd.DataFrame.from_dict(data)

    elif type == 'tp_slope':
        data = {}
        for i in range(len(list_of_countries_considered)):
            data.update({list_of_countries_considered[i]:globals()[list_of_countries_considered[i]]['tp_slope']})
            if min(globals()[list_of_countries_considered[i]]['tp_slope']) < min_value:
                min_value = min(globals()[list_of_countries_considered[i]]['tp_slope'])
            if max(globals()[list_of_countries_considered[i]]['tp_slope']) > max_value:
                max_value = max(globals()[list_of_countries_considered[i]]['tp_slope'])
        df=pd.DataFrame.from_dict(data)

    elif type == 'tp_curvature':
        data = {}
        for i in range(len(list_of_countries_considered)):
            data.update({list_of_countries_considered[i]:globals()[list_of_countries_considered[i]]['tp_curvature']})
            if min(globals()[list_of_countries_considered[i]]['tp_curvature']) < min_value:
                min_value = min(globals()[list_of_countries_considered[i]]['tp_curvature'])
            if max(globals()[list_of_countries_considered[i]]['tp_curvature']) > max_value:
                max_value = max(globals()[list_of_countries_considered[i]]['tp_curvature'])
        df=pd.DataFrame.from_dict(data)

    else:
        data = {}
        for i in range(len(list_of_countries_considered)):
            data.update({list_of_countries_considered[i]:globals()[list_of_countries_considered[i]]['er_return']})
            if min(globals()[list_of_countries_considered[i]]['er_return']) < min_value:
                min_value = min(globals()[list_of_countries_considered[i]]['er_return'])
            if max(globals()[list_of_countries_considered[i]]['er_return']) > max_value:
                max_value = max(globals()[list_of_countries_considered[i]]['er_return'])
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

        # Same limits for everybody!
        plt.ylim(min_value,max_value)

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


x = plot_returns("One Month Ahead Foreign Exchange Return", 'ylds_level')

max(C.ylds_level)


palette = plt.get_cmap('Set1')
palette(7)
