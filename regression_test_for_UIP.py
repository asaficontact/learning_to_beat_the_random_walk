from imports import *
from functions_kit import *
%matplotlib qt


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


#Get Exchange Rates from Fred
api = '3dcbccc26181a5457fb8bd0584de00a8'
exchange_rates = get_exchange_rates(api)
exchange_rates.tail()


y = exchange_rates['UK']
y = np.log(y).diff()
interest_UK = ylds_UK['V2']
interest_US = ylds_US['V2']
x = interest_US - interest_UK
x.index = x.index + pd.offsets.MonthBegin(0) #Change to beginning of month from end of month
x.dropna(inplace = True)
y.dropna(inplace = True)
x.head()
y.head()
start_date = y.index[0]
x.tail()

y.tail()
end_date = x.index[-1]

y = y[start_date:end_date]
x = x[start_date:end_date]

y.head()
x.head()

len(y)
len(x)
import statsmodels.api as sm
X = sm.add_constant(x) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
model.summary()

import statsmodels.formula.api as smf
d = { "x": pd.Series(x), "y": pd.Series(y)}
df = pd.DataFrame(d)
mod = smf.ols('y ~ x', data=df).fit()
print(mod.summary())


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

f = open('current_reg.tex', 'w')
f.write(beginningtex)
f.write(mod.summary().as_latex())
f.write(endtex)
f.close()

%matplotlib inline
plt.scatter(x,y,color='#003F72',label='data')
plt.plot(x, predictions, label='regression line', color = 'red')
# plt.ylabel("Change in GBP/USD")
# plt.xlabel("Interest Rate Differential")
plt.legend(loc=1)
plt.savefig('images/reg/current.png')


#6 months ahead

y = exchange_rates['UK']
y = np.log(y).diff()
interest_UK = ylds_UK['V2']
interest_US = ylds_US['V2']
x = interest_US - interest_UK
x.index = x.index + pd.offsets.MonthBegin(0) #Change to beginning of month from end of month
x.dropna(inplace = True)
y.dropna(inplace = True)
x.head()

y.head()

start_x = x[:'1999-02-01'][-6:].index[0]
y.tail()
end_x = x[:'2019-09-01'][-3:].index[0]

x_six = x[start_x:end_x]
x_six.reset_index(drop = True, inplace = True)
y_six = y.copy()
y_six.reset_index(drop = True, inplace = True)
len(x_six)
len(y_six)


d = { "x": pd.Series(x_six), "y": pd.Series(y_six)}
df_six = pd.DataFrame(d)
mod_six = smf.ols('y ~ x', data=df_six).fit()
print(mod_six.summary())


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

f = open('current_reg_six.tex', 'w')
f.write(beginningtex)
f.write(mod_six.summary().as_latex())
f.write(endtex)
f.close()


predictions = mod_six.predict(df_six['x'])
plt.scatter(x_six,y_six,color='#003F72',label='data')
plt.plot(x_six, predictions, label='regression line', color = 'red')
plt.legend(loc=1)
plt.savefig('images/reg/current_six.png')

plt.show()

#1 Year ahead

y = exchange_rates['UK']
y = np.log(y).diff()
interest_UK = ylds_UK['V2']
interest_US = ylds_US['V2']
x = interest_US - interest_UK
x.index = x.index + pd.offsets.MonthBegin(0) #Change to beginning of month from end of month
x.dropna(inplace = True)
y.dropna(inplace = True)
x.head()

y.head()

start_x = x[:'1999-02-01'][-12:].index[0]
y.tail()
end_x = x[:'2019-09-01'][-9:].index[0]

x_12 = x[start_x:end_x]
x_12.reset_index(drop = True, inplace = True)
y_12 = y.copy()
y_12.reset_index(drop = True, inplace = True)
len(x_12)
len(y_12)


d = { "x": pd.Series(x_12), "y": pd.Series(y_12)}
df_12 = pd.DataFrame(d)
mod_12 = smf.ols('y ~ x', data=df_12).fit()
print(mod_12.summary())

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

f = open('current_reg_12.tex', 'w')
f.write(beginningtex)
f.write(mod_12.summary().as_latex())
f.write(endtex)
f.close()

predictions = mod_12.predict(df_12['x'])
plt.scatter(x_12,y_12,color='#003F72',label='data')
plt.plot(x_12, predictions, label='regression line', color = 'red')
plt.legend(loc=1)
plt.savefig('images/reg/current_12.png')
plt.show()
