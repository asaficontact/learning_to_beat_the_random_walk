# Learning to Beat the Random Walk - Using Machine Learning to Predict Changes in Exchange Rates

In my thesis, I use different machine learning techniques to predict the directional change in exchange rates. I start off by analyzing Uncovered Interest Rate Parity (UIP) and its failure to predict changes in exchange rates. Using linear regression, I show that the β coefficient in UIP equation is not equal to zero over the short and long run. This shows the importance of currency risk premium for understanding changes in exchange rates. However, risk premium and market expectations are extremely difficult to measure. For this reason, Random Walk is the best model for predicting changes in the foreign exchange rates over the short run. This lead me to ask: Can we use the latest machine learning techniques to predict foreign exchange rates more accurately than Random Walk model? I explore various machine learning techniques including Principal Component Analysis (PCA), Support Vector Machines (SVM), Artificial Neural Networks (ANN), and Sentiment Analysis in an effort to predict the directional changes in exchange rates for a list of developed and developing countries.

After exploring relevant literature on exchange rates, I analyze excess returns in the carry trade market by using historical exchange rate and interest rate data to create a simulation for trading on a monthly or semi-annual basis. I use the interest rate data of each country to sort them into portfolios for carry trade. The results show that investors can earn large profits over the short run by borrowing from low interest rate countries’ bond market and investing in that of high interest rate countries. I also find that the returns on carry trade starts to reduce as the trading interval increases.

According to the spanning hypothesis, the yield curve, its expectation and term premium component span all relevant information related to the economic performance of a country. This means that the economic situation of a country can be understood by analyzing its bond market. Based on the spanning hypothesis, I use Principal Component Analysis (PCA) to extract the level, slope, and curvature of the yield curve and its components. Then, taking the resultant principal components, I use Support Vector Machines (SVM) and Artificial Neural Networks (ANN) to predict the directional change in foreign exchange excess returns. Compared to the SVMs, the ANNs more accurately predict the directional change in the exchange rate excess returns over the short run. This difference is attributed to the ability of Artificial Neural Networks to learn abstract features from raw data.

Finally, in the last chapter, I use sentiment analysis to evaluate the tonality of foreign exchange news articles from investing.com to develop a variable for understanding currency risk premium and market expectations in order to predict the directional changes in the exchange rates. In order to identify the model that most reliably captures the tonality of foreign exchange news articles, I use five different sentiment analysis models: Tonality model, VADER model, TextBlob model, Harvard IV-4 Dictionary model, and Loughran and Mcdonald Dictionary model. The results show that different sentiment models better predict the directional changes in exchange rates of different countries.

Overall, this research finds evidence that machine learning models can more accurately predict directional change in exchange rates than the Random Walk model when the input features used to train the machine learning models are representative of the economic situation of a country. Hence, using macro-economic and financial theory to develop input features that capture the most relevant characteristics of a country’s economy can help us understand short run movements in the foreign exchange markets.

For Full Paper: [CLICK HERE](https://github.com/asaficontact/learning_to_beat_the_random_walk/blob/master/thesis.pdf)
For Sentiment Analysis App: [CLICK HERE](https://asaficontact.shinyapps.io/fx_sentiment/)

## Understanding the Code Files
* **main.py** contains the code for carry trade portfolio simulation and predicting directional change in exchange rate returns using Principal Component Analysis, Support Vector Machines and Artifical Neural Networks. 
* **carryTrader.py** contains the code for carry trade portfolio simulation class. 
* **forexTrader.py** contains the code for predicting directional change in exchange rate returns class. 
* **functions_kit.py** contains the code for all functions used in main.py
* **regression_test_for_UIP.py** contains the code running Uncovered Interest Rate Parity regressions. 
* **thesis.tex** contains the latex code file used to write my thesis. 

### Sentiment_Analysis_Code_Files folder contains the following code files that were used for sentiment analysis: 
* **sentiment_analysis_main.py** contains the code for preprocessing daily news articles dataset, sorting them by country, and then applying five different sentiment models for tonality extraction to predict directional change in exchange rates. 
* **function_kit.py** contains the code for all functions used in sentiment_analysis_main.py
* **sentiment_analyzer.py** contains the code for a class that calculates tonality score using five different sentiment models. 
* **tfidf_neural.py** contains the code for a class that uses TFIDF vectorization along with a multi-layer perceptron to train models for predicting directional change in exchange rates using foreign exchange news paper articles dataset. 

## Sentiment Analysis Dataset
The financial news articles for our analysis were collected from investing.com using web scraping. Web scraping is technique used for extracting data from a website. The textual dataset consists of all foreign exchange news articles from 2009 to 2019 recored on daily frequency. The total dataset consists of about 82,500 news articles. If anyone is interested in using the dataset for their own project please contact at: asaficontact@gmail.com


