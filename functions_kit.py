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
    print(pca.explained_variance_ratio_)
    return principalDf


#Merge Datasets
def merge_datasets(ylds_df, exp_df, tp_df, exchange_rates_df):
    data_frames = [ylds_df, exp_df, tp_df, exchange_rates_df]
    result = reduce(lambda  left,right: pd.merge(left,right,left_index = True, right_index = True,
                                                how='inner'), data_frames)
    return result


#Calculate Exchange Rate returns
def calculate_returns(df, country_name, nlag = 1):
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
