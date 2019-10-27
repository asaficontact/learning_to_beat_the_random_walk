from imports import *
from functions_kit import *

class ct:
    def __init__(self, budget, nlag, load = True):
        self.budget = budget
        self.nlag = nlag
        self.df, self.date_list = self.prepareDataset(load)


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

        #For Carry Trade
        if self.nlag > 3:
            lag_value = self.nlag
        else:
            lag_value = 3
        for i in range(len(list_of_countries)):
            globals()['train_ylds_' + list_of_countries[i]]['shortRun_interest'] = globals()['ylds_' + list_of_countries[i]].iloc[:training_length + 1]['V' + str(lag_value)]
            globals()['test_ylds_' + list_of_countries[i]]['shortRun_interest'] = globals()['ylds_' + list_of_countries[i]].iloc[training_length:]['V' + str(lag_value)]

        #US Dataset
        train_ylds_US = PCA_analysis(ylds_US.iloc[:training_length + 1], 'ylds', False)
        train_exp_US = PCA_analysis(exp_US.iloc[:training_length + 1], 'exp', False)
        train_tp_US = PCA_analysis(tp_US.iloc[:training_length + 1], 'tp', False)

        test_ylds_US = PCA_analysis(ylds_US.iloc[training_length:], 'ylds', False)
        test_exp_US = PCA_analysis(exp_US.iloc[training_length:], 'exp', False)
        test_tp_US = PCA_analysis(tp_US.iloc[training_length:], 'tp', False)

        #US for Carry Trade
        train_ylds_US['shortRun_interest'] = ylds_US.iloc[:training_length + 1]['V' + str(lag_value)]
        test_ylds_US['shortRun_interest'] = ylds_US.iloc[training_length:]['V' + str(lag_value)]

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

        #Calculate Differential
        for i in range(len(list_of_countries_considered)):
            globals()["train_" + list_of_countries_considered[i]] = calculate_differential(globals()["train_" + list_of_countries_considered[i]], train_US,'carry')
            globals()["test_" + list_of_countries_considered[i]] = calculate_differential(globals()["test_" + list_of_countries_considered[i]], test_US,'carry')

        #Rename column
        for i in range(len(list_of_countries_considered)):
            globals()["train_" + list_of_countries_considered[i]].rename(columns={list_of_countries_considered[i]:'exchange_rate'}, inplace=True)
            globals()["test_" + list_of_countries_considered[i]].rename(columns={list_of_countries_considered[i]:'exchange_rate'}, inplace=True)

        #concat dataframes
        for i in range(len(list_of_countries_considered)):
            globals()[list_of_countries_considered[i]] = pd.concat([globals()["test_" + list_of_countries_considered[i]][1:],globals()["train_" + list_of_countries_considered[i]]])
            globals()[list_of_countries_considered[i]] = globals()[list_of_countries_considered[i]].sort_index()

        #move exchange rates
        for i in range(len(list_of_countries_considered)):
            globals()[list_of_countries_considered[i]]['forward_er'] = globals()[list_of_countries_considered[i]]['exchange_rate'].shift(-self.nlag)
            globals()[list_of_countries_considered[i]] = globals()[list_of_countries_considered[i]].iloc[:-self.nlag]

        #Update Date list
        date_list = list(AUS.index)

        #Create a multi level index:
        list_of_countries_variables = [AUS, CAN, CHI, CHN, COL, CZR, EUR, HUN,
                            INDO, JAP, MEX, NOR, NZ, PO,SA, SNG,SWE, SWI, UK ,BRA]
        carry_trade_data = pd.concat(list_of_countries_variables, keys = list_of_countries_considered, axis = 0)
        carry_trade_data = carry_trade_data.swaplevel(-2,-1).sort_index()

        return carry_trade_data, date_list


    def basic_trade(self, weights = [100] , reinvest = False):

        total_profit = []
        total_percentage_profit = []
        borrow_list = []
        repay_list = []
        invest_list = []
        interest_list = []
        earning_list = []
        repayment_list = []
        profit_list = []
        percentage_profit_list = []
        short_country_name_list = []
        long_country_name_list = []

        initial_budget = self.budget
        for j in range(len(self.date_list)-1):
            trade_order = self.df.xs(self.date_list[j]).sort_values('shortRun_interest_diff')
            country_list = list(trade_order.index)

            short_country = []
            short_country_name = []
            long_country = []
            long_country_name = []
            short_country_index = 0
            long_country_index = -1

            for weight in weights:
                short_country.append(trade_order.iloc[short_country_index])
                short_country_name.append(country_list[short_country_index])

                long_country.append(trade_order.iloc[long_country_index])
                long_country_name.append(country_list[long_country_index])

                short_country_index += 1
                long_country_index -= 1

            short_borrow_amount = []
            long_invest_amount = []
            short_repay_amount = []
            long_interest_amount = []
            long_earning = []
            short_repayment = []
            profit = []
            percentage_profit = []

            for i in range(len(weights)):
                short_borrow_amount.append(short_country[i]['exchange_rate'] * (initial_budget * (weights[i]/100)))
                short_repay_amount.append(short_borrow_amount[i] * (1 + (short_country[i]['shortRun_interest']/100)))

                long_invest_amount.append(long_country[i]['exchange_rate'] * (initial_budget * (weights[i]/100)))
                long_interest_amount.append(long_invest_amount[i] * (1 + (long_country[i]['shortRun_interest']/100)))

                long_earning.append(long_interest_amount[i] / long_country[i]['forward_er'])
                short_repayment.append(short_repay_amount[i] / short_country[i]['forward_er'])

                profit.append(long_earning[i] - short_repayment[i])
                percentage_profit.append(((long_earning[i] - short_repayment[i])/initial_budget)*100)

            total_borrow_amount = sum(short_borrow_amount)
            total_repay_amount = sum(short_repay_amount)
            total_invest_amount = sum(long_invest_amount)
            total_interest_amount = sum(long_interest_amount)
            total_earning = sum(long_earning)
            total_repayment = sum(short_repayment)
            total_profit.append(total_earning - total_repayment)
            total_percentage_profit.append(((total_earning - total_repayment)/initial_budget)*100)

            borrow_list.append(short_borrow_amount)
            repay_list.append(short_repay_amount)
            invest_list.append(long_invest_amount)
            interest_list.append(long_interest_amount)
            earning_list.append(long_earning)
            repayment_list.append(short_repayment)
            profit_list.append(profit)
            percentage_profit_list.append(percentage_profit)
            short_country_name_list.append(short_country_name)
            long_country_name_list.append(long_country_name)

            if reinvest:
                initial_budget += total_profit[j]

        result = { "short_country_name_list": short_country_name_list,
                   "long_country_name_list": long_country_name_list,
                   "borrow_list": borrow_list,
                   "repay_list": repay_list,
                   "invest_list": invest_list,
                   "interest_list": interest_list,
                   "earning_list": earning_list,
                   "repayment_list": repayment_list,
                   "profit_list": profit_list,
                   "percentage_profit_list": percentage_profit_list,
                   "average_profit": mean(total_profit),
                   "average_percentage_profit": mean(total_percentage_profit),
        }

        total_profit = pd.DataFrame(total_profit, index = self.date_list[:-1], columns = ['profit'])
        total_percentage_profit = pd.DataFrame(total_percentage_profit, index = self.date_list[:-1], columns = ['percentage_profit'])

        return total_profit, total_percentage_profit, result
