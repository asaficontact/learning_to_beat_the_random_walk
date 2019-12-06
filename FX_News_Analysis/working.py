from imports import *
from functions_kit import *
%matplotlib qt

##################################
#Process Data to save it as pickle
##################################

#Load news Paper Articles dataset
with open('data/total.json') as json_file:
    data = json.load(json_file)

with open('data/2008-2009.json') as json_file:
    data1 = json.load(json_file)


df_load1 = pd.DataFrame.from_dict(data)
df_load2 = pd.DataFrame.from_dict(data1)
df_load = pd.concat([df_load1, df_load2], join = 'inner', ignore_index = True)

len(df_load1)
len(df_load2)
len(df_load)
df_load.head()

#Drop unwanted columns and change Unix time stamp to datetime object
df = df_load.drop(['end', 'article_link'], axis = 1)
df['date'] = pd.to_datetime(df['timestamp'], unit = 's')
df = df.drop(['timestamp'], axis = 1)
df.set_index('date', drop = True, inplace = True)
df.sort_index(inplace = True)

#Check for NA
df.isna().sum()
df.dropna(inplace = True)
df.isna().sum()


df_load.duplicated().sum()
df_load.drop_duplicates(keep = 'first', inplace = True)
#Check for duplicates:
df.duplicated().sum()
df[df.duplicated(keep = False)]
df.drop_duplicates(keep = 'first', inplace = True)

df.duplicated().sum()

len(df)

#Save cleaned data as pickle
df.to_pickle('data/complete_data.pkl')


############################
#Load Processed DataFrame
############################


df = pd.read_pickle('data/complete_data.pkl')

df.head()


def count_news_frequency(df, plot_data = False):
    date_list = [str(date)[0:10] for date in list(df.index)]
    #Get rid of duplicates:
    date_list = list(dict.fromkeys(date_list))

    date_count = []
    for date in date_list:
        date_count.append(len(df[date]))

    news_release_rate = {"date": date_list, "number_of_news_releases": date_count}
    news_release_rate = pd.DataFrame.from_dict(news_release_rate)
    news_release_rate['date'] = pd.to_datetime(news_release_rate['date'])
    news_release_rate.set_index('date', drop = True, inplace = True)

    if plot_data:
        plt.title("")
        plt.ylabel('Number of News Releases')
        plt.xlabel('Date')
        plt.bar(news_release_rate.index, news_release_rate.number_of_news_releases, color = 'r')
        # plt.savefig('news_frequency.png')
    return news_release_rate


#Count number of news per day and visualize it:
news_frequency = count_news_frequency(final_df['CAN'], plot_data = True)

news_frequency = count_news_frequency(final_df['CHN'], plot_data = True)

news_frequency = count_news_frequency(final_df['JAP'], plot_data = True)

news_frequency = count_news_frequency(final_df['AUS'], plot_data = True)



################################
#Basic Text Cleanup:
################################

def remove_HTMLtags(text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)


def Punctuation(string):

    # punctuation marks
    punctuations = '''!()-[]{};:'"\,<>./?@#\n$%^&*_~\n\n'''

    # traverse the given string and if any punctuation
    # marks occur replace it with null
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, " ")

    # Print string without punctuation
    return string

def regex_cleanUp(text_doc):

        result = []
        for text in text_doc:
            x = remove_HTMLtags(text)
            result.append(x.lower())
        len(result)
        text_doc = result



        result = []
        for text in tqdm.tqdm(text_doc):
            x = Punctuation(text)
            result.append(x.lower())
        len(result)
        text_doc = result

        return text_doc

#Clean up text
df['text'] = regex_cleanUp(df.text)
#Clean up title
df['title'] = regex_cleanUp(df.title)

df.head()
#####################################
#Spacy Text Cleanup
####################################
import spacy
nlp = spacy.load("en_core_web_lg")

def keep_token(t):
        return (t.is_alpha and
                not (t.is_space or t.is_punct or
                     t.is_stop or t.like_num))


def lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if keep_token(t)]


def spacy_cleanUp(text_doc):
        doc_text = []
        for text in tqdm.tqdm(text_doc):
            doc_text.append(nlp.tokenizer(text))

        result = []
        for i in tqdm.tqdm(range(0, len(doc_text))):
            doc = lemmatize_doc(doc_text[i])
            result.append(doc)

        doc_text = result

        result = []
        for doc in doc_text:
            result_doc = ' '.join(doc)
            result.append(result_doc)

        doc_text = result

        return doc_text


df['text'] = spacy_cleanUp(df.text)

df['title'] = spacy_cleanUp(df.title)

#Get rid of basic a,b,c,d,... alphabets
df['text'] = df.text.map(lambda x: ' '.join( [w for w in x.split() if len(w)>1] ))
df['title'] = df.title.map(lambda x: ' '.join( [w for w in x.split() if len(w)>1] ))

#Save the completely Cleaned File:
df.to_pickle('data/final_complete_data.pkl')



################################
#Load Fully Processed DataFrame
################################

df = pd.read_pickle('data/final_complete_data.pkl')

df.iloc[0].text

def word_counter(df, list_, type = 'or'):        #CHECK THIS FUNCTION SEEMS TO BE INCORRECT
    result = [0] * len(df.text)
    for i, text, title in zip(range(len(df.text)),df.text, df.title):
        if type == 'or':
            for word in list_:
                if word in text.lower() or word in title.lower():
                    result[i] = 1
                    break
        else:
            for word in list_:
                and_counter = 0
                if word in text.lower() or word in title.lower():
                    and_counter = 1
                else:
                    break
            if and_counter == 1:
                result[i] = 1
    return result

def df_filter(df, list_, type_ = 'or'):
    index = df.index
    index_list = []
    text_list = []
    title_list = []
    for i, text, title in tqdm.tqdm(zip(range(len(df.text)), df.text, df.title)):
        if type_ == 'or':
            for word in list_:
                if word in text.lower() or word in title.lower():
                    text_list.append(text)
                    title_list.append(title)
                    index_list.append(index[i])
                    break
        else:
            for word in list_:
                and_counter = 0
                if word in text.lower() or word in title.lower():
                    and_counter = 1
                else:
                    break
            if and_counter == 1:
                text_list.append(text)
                title_list.append(title)
                index_list.append(index[i])

    result = pd.DataFrame({'date': index_list, 'text': text_list, 'title': title_list})
    result.set_index('date', drop = True, inplace = True)
    result.sort_index(inplace = True)
    return result


x = df_filter(df, ['london', 'united kingdom', 'uk', 'pound', 'england', 'britain'], type = 'or')
x = df_filter(x, [], type = 'or')
len(x)
x.head()

###################################
#Develop Model For Cosine Similarity
##################################
#Vectorize data
doc_text = []
for text in tqdm.tqdm(df.text):
    doc_text.append(nlp(text))
doc_title = []
for title in tqdm.tqdm(df.title):
    doc_title.append(nlp(title))

def remove_pronoun(doc):
    result_list = []
    for text in tqdm.tqdm(doc):
        result = [str(token) for token in text if token.lemma_ != '-PRON-']
        result = ' '.join(result)
        result_list.append(result)
    return result_list

df['text'] = remove_pronoun(doc_text)
df['title'] = remove_pronoun(doc_title)

# index_list = list(df.index)
# df_nlp = pd.DataFrame({'date': index_list, 'text': doc_text, 'title': doc_title})
# df_nlp.set_index('date', drop = True, inplace = True)
# df_nlp.sort_index(inplace = True)

#Create search terms for direct search
countries_search_terms = {'AUS': ['australia', 'aud', 'australian'],
                            'CAN': ['canada', 'cad', 'canadian'],
                            'CHI': ['chile', 'clp', 'chilean'],
                            'CHN': ['china', 'rmb', 'renminbi', 'yuan', 'chinese'],
                            'COL': ['colombia', 'cop', 'colombian'],
                            'CZR': ['czech republic', 'czk', 'prague'],
                            'EUR': ['euro', 'eur', 'germany', 'italy', 'greece', 'eu', 'brussels'],
                            'HUN': ['hungary', 'huf', 'forint', 'hungarian'],
                            'INDO': ['indonesia', 'idr', 'rupiah', 'indonesian'],
                            'JAP': ['japan', 'jpy', 'jp', 'yen'],
                            'MEX': ['mexico', 'mxn', 'mexican'],
                            'NOR': ['norway', 'nok', 'norwegian'],
                            'NZ': ['zealand', 'nzd', 'kiwi'],
                            'PO': ['poland', 'pln', 'zloty', 'polish'],
                            'SA': ['africa', 'zar', 'rand', 'african'],
                            'SNG': ['singapore', 'sgd'],
                            'SWE': ['sweden', 'sek', 'swedish'],
                            'SWI': ['switzerland', 'chf', 'franc', 'swiss'],
                            'UK':  ['britain', 'gbp', 'pound', 'british'],
                            'BRA': ['brazil', 'brl', 'brazilian']}

# countries_search_terms["AUS"][0:2]
# x = nlp('yen')
# x.similarity(doc_text[0])
# x.similarity(doc_text[1])
# x.similarity(doc_text[2])
# x.similarity(doc_text[3])

# 'stock' in df['text'][1]
# y = ['Hello', 'World', 'what is goin']

def ultimate_filter(df, doc_text, doc_title, terms):
    simple_result = df_filter(df, terms, type_ = 'or')
    advance_result = consine_similarity_filter(df, doc_text, doc_title, terms[0:2])
    final_result = pd.concat([simple_result,advance_result]).drop_duplicates().sort_index()
    # pd.merge(simple_result,advance_result,left_index = True, right_index = True,
    #                                             how='outer')
    return final_result

def consine_similarity_filter(df, doc_text, doc_title, word_list):
    search_list = []
    for word in word_list:
        search_list.append(nlp(word))
    index = df.index
    index_list = []
    text_list = []
    title_list = []
    for i, text, title in tqdm.tqdm(zip(range(len(df.text)), doc_text, doc_title)):
        for search_string in search_list:
            if search_string.similarity(text) > 0.5 or search_string.similarity(title) > 0.5:
                text_list.append(str(text))
                title_list.append(str(title))
                index_list.append(index[i])
                break
    result = pd.DataFrame({'date': index_list, 'text': text_list, 'title': title_list})
    result.set_index('date', drop = True, inplace = True)
    result.sort_index(inplace = True)
    return result


list_of_countries_considered = ["AUS", "CAN", "CHI", "CHN", "COL", "CZR", "EUR", "HUN",
                    "INDO", "JAP", "MEX", "NOR", "NZ", "PO","SA", "SNG","SWE", "SWI", "UK" ,"BRA"]


for i in range(len(list_of_countries_considered)):
    globals()['df_' + list_of_countries_considered[i]] = ultimate_filter(df, doc_text, doc_title, countries_search_terms[list_of_countries_considered[i]])
    globals()['df_' + list_of_countries_considered[i]].to_pickle('news_data/news_' + list_of_countries_considered[i] + '.pkl')

df_AUS = ultimate_filter(df, doc_text, doc_title, countries_search_terms['AUS'])
df_AUS.head()
len(df_AUS)
len(df_EUR)
len(df_CAN)
len(df_BRA)
df_AUS.duplicated().sum()

df.head()

###################################
#Load Country Specific News Data
###################################
list_of_countries_considered = ["AUS", "CAN", "CHI", "CHN", "COL", "CZR", "EUR", "HUN",
                    "INDO", "JAP", "MEX", "NOR", "NZ", "PO","SA", "SNG","SWE", "SWI", "UK" ,"BRA"]

df_countries = {}
for country in list_of_countries_considered:
    df_countries.update({country: pd.read_pickle('news_data/news_' + country + '.pkl')})

#Check for length of news data in every country:
df_countries_length = {}
for country in list_of_countries_considered:
    df_countries_length.update({country: len(df_countries[country])})

df_countries_length

#Keep articles that has length greater than 3000:

final_list_of_countries = []
for country in list_of_countries_considered:
    if df_countries_length[country] > 3000:
        final_list_of_countries.append(country)

final_list_of_countries

#Use the final list to create a final dataframe dictionary

final_df = {}
for country in final_list_of_countries:
    final_df.update({country: df_countries[country]})

#Check for length of news data in every country:
final_df_length = {}
for country in final_list_of_countries:
    final_df_length.update({country: len(final_df[country])})

final_df_length

####################################
#Get Daily Exchange Rate Datasets
###################################
dates = list(df.index)
list_of_dates = []
for date in tqdm.tqdm(dates):
    if str(date)[0:10] not in list_of_dates:
        list_of_dates.append(str(date)[0:10])

len(list_of_dates)


list_of_dates[0:5]

#Make API requests to openexchangerates
timestamp_list = []
exchange_rates = []
result = []
for date in tqdm.tqdm(list_of_dates):
    response = requests.get("https://openexchangerates.org/api/historical/" + date + ".json?app_id=f26fd948b3584ed6b75e2d9c8ca2c4a2")
    result.append(response.json())

#Extract timestamp and rates from results recieved

for data in result:
    timestamp_list.append(data['timestamp'])
    exchange_rates.append(data['rates'])

#Create a currency list
final_list_of_countries
final_list_of_currencies = ['AUD', 'CAD', 'CNY', 'COP', 'EUR', 'JPY', 'NZD', 'ZAR', 'SGD', 'CHF', 'GBP']

exchange_rates[0]

#Create currency lists
for currency in final_list_of_currencies:
    globals()[currency] = []

#Add currencies to specific lists
for exchange_rate in tqdm.tqdm(exchange_rates):
    for currency in final_list_of_currencies:
        globals()[currency].append(exchange_rate[currency])
len(AUD)
len(EUR)
len(timestamp_list)

#Compile all Exchange rate data in a dataframe format
exchange_rate_data = {'timestamp': timestamp_list}
for currency in tqdm.tqdm(final_list_of_currencies):
    exchange_rate_data.update({currency: globals()[currency]})

#Store all the recieved data in dataframe format
exchange_rate_df = pd.DataFrame.from_dict(exchange_rate_data)
exchange_rate_df['date'] = pd.to_datetime(exchange_rate_df['timestamp'], unit = 's')
exchange_rate_df = exchange_rate_df.drop(['timestamp'], axis = 1)
exchange_rate_df.set_index('date', drop = True, inplace = True)
exchange_rate_df.sort_index(inplace = True)
exchange_rate_df.drop_duplicates(keep = 'first', inplace = True)
len(exchange_rate_df)
exchange_rate_df.head(-10)

#Save exchange rate data
exchange_rate_df.to_pickle('data/final_exchange_rate_daily_data.pkl')


################################
#Load Exchange Rate Dataset
################################

exchange_rate_df = pd.read_pickle('data/final_exchange_rate_daily_data.pkl')
len(exchange_rate_df)

#Reduce datetime to daily for the dataset
exchange_rate_df.index = exchange_rate_df.index.values.astype('<M8[D]')

for country in final_list_of_countries:
    final_df[country].index = final_df[country].index.values.astype('<M8[D]')





exchange_rate_df.head()


final_df['CAN'].tail(10)

#Combine articles as per daily basis
for country in tqdm.tqdm(final_list_of_countries):
    final_df[country] = final_df[country].groupby([final_df[country].index])['text', 'title'].agg(lambda x: ' '.join(x.astype(str)))


final_df['CAN'].tail(10)
#Get UK exchange rates
UK_exchange = exchange_rate_df['AUD']
UK_change = UK_exchange.pct_change()
UK_change.dropna(inplace = True)

model_data = pd.concat([final_df['AUS'], UK_change], axis = 1)
model_data.dropna(inplace = True)
len(model_data)

model_data.head()
#Get positive and negative change binary column:
binary_column = [1 if value > 0 else 0 for value in model_data['AUD']]
model_data['change'] = binary_column

##### Setting up the tfidf model #############################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import regularizers
lb = LabelBinarizer(sparse_output = True)    #Probably dont need it as labels are already binary

tfidf = TfidfVectorizer(lowercase = True, max_df = 0.95, stop_words='english')

x = model_data[['text', 'title']]
y = model_data['change']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 42)

dtm_train = tfidf.fit_transform(x_train['text'] + x_train['title'])
dtm_train



#Creating Neural Network
tag_classifier = Sequential()
#first layer
tag_classifier.add(Dense(512, activation='relu', kernel_initializer='random_normal', input_dim=79158, kernel_regularizer=regularizers.l2(0.01)))
tag_classifier.add(Dropout(.2))
#second layer
tag_classifier.add(Dense(64, activation='relu', kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(0.01)))
tag_classifier.add(Dropout(.2))
#output layer
#softmax sums predictions to 1, good for multi-classification
tag_classifier.add(Dense(1, activation ='sigmoid', kernel_initializer='random_normal'))
tag_classifier.summary()

#Compiling
#adam optimizer adjusts learning rate throughout training
#loss function categorical crossentroy for classification
tag_classifier.compile(optimizer ='adam',loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stop = EarlyStopping(monitor = 'val_err', patience = 5, verbose = 2)

tag_classifier.fit(dtm_train, y_train, epochs = 500,
          batch_size = 10000, verbose = 2,
          callbacks = [early_stop])

tag_classifier.save('news_data/tfidf_neural_model.h5')

#Load the neural network:
tag_classifier = load_model('news_data/tfidf_neural_model.h5')

#Get predictions on test data
dtm_test = tfidf.transform(x_test['text'] + x_test['title'])
y_pred_test = tag_classifier.predict(dtm_test)
y_predictions = lb.inverse_transform(Y = y_pred_test, threshold=0.5)

cm = confusion_matrix(y_test, y_predictions)
c_report = classification_report(y_test, y_predictions)
accuracy_score(y_test, y_predictions)
cm

#Get predictions for train data
y_pred = tag_classifier.predict(dtm_train)
y_predictions = lb.inverse_transform(Y = y_pred, threshold=0.5)
accuracy_score(y_train, y_predictions)
cm = confusion_matrix(y_train, y_predictions)
cm

from tfidf_neural import tn
x_train_list = {}
x_test_list = {}
y_train_list = {}
y_test_list = {}
model_list = {}
final_list_of_currencies = list(exchange_rate_df.columns)
final_list_of_currencies
for country, currency in tqdm.tqdm(zip(final_list_of_countries, final_list_of_currencies)):
    x = tn(final_df[country])
    x_train, x_test, y_train, y_test = x.prepare_data(exchange_rate_df, currency)
    model = x.setup_neural_network(x_train, x_test, y_train, y_test, currency, 0.06)
    x_train_list.update({country: x_train})
    x_test_list.update({country: x_test})
    y_train_list.update({country: y_train})
    y_test_list.update({country: y_test})
    model_list.update({country: model})



x = tn(final_df['CAN'])
x_train, x_test, y_train, y_test = x.prepare_data(exchange_rate_df, 'CAD')
CAD_model = x.setup_neural_network(x_train, x_test, y_train, y_test, 'CAD', 0.08)
AUS.model

#Get predictions on test data
dtm_test = tfidf.transform(x_test['text'] + x_test['title'])
y_pred_test = CAD_model.predict(dtm_test)
y_pred_test[0][0]
y_predictions = [1 if value[0] >= 0.5 else 0 for value in y_pred_test]
accuracy_score(y_test, y_predictions)
cm = confusion_matrix(y_test, y_predictions)
c_report = classification_report(y_test, y_predictions)
cm
len(y_predictions)
len(y_pred_train)
dtm_train = tfidf.transform(x_train['text'] + x_train['title'])
y_pred_train = AUS_model.predict(dtm_train)
y_pred_test[0][0]
y_predictions = [1 if value[0] >= 0.5 else 0 for value in y_pred_train]
accuracy_score(y_train, y_predictions)
cm = confusion_matrix(y_train, y_predictions)
c_report = classification_report(y_test, y_predictions)
cm
##########################
# Sentiment analysis
##########################

import gensim
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full
from gensim import models

#Vectorize data and add title and text
doc_text = []
for text in tqdm.tqdm(final_df['UK'].text + final_df['UK'].title):
    doc_text.append(nlp(text))
# doc_title = []
# for title in tqdm.tqdm(final_df['UK'].title):
#     doc_title.append(nlp(title))

# Clean the text
token_tot_text = list()
for i in tqdm.tqdm(range(len(doc_text))):
    tokens_i = [token.text.lower() for token in doc_text[i] if not token.is_stop]
    token_tot_text.append(tokens_i)

# token_tot_title = list()
# for i in tqdm.tqdm(range(len(doc_title))):
#     tokens_i = [token.text.lower() for token in doc_title[i] if not token.is_stop]
#     token_tot_title.append(tokens_i)

# Dictionary of the whole corpus
#dataset = [d.split() for d in token_tot]
dictionary_text = Dictionary(token_tot_text)
# dictionary_title = Dictionary(token_tot_title)

#The corpus
docs_corpus_text = [dictionary_text.doc2bow(doc) for doc in token_tot_text]
# docs_corpus_title = [dictionary_title.doc2bow(doc) for doc in token_tot_title]

# Show the Word Weights in Corpus
for doc in docs_corpus_text:
   print([[dictionary_text[id], freq] for id, freq in doc])

# Create the TF-IDF model
tfidf_text = models.TfidfModel(docs_corpus_text, smartirs='ntc')
# tfidf_title = models.TfidfModel(docs_corpus_title, smartirs='ntc')

# Show the TF-IDF weights
for doc in tfidf_text[docs_corpus_text]:
    print([[dictionary_text[id], np.around(freq, decimals=3)] for id, freq in doc])

#Construct negative dictionary

texts_org = ' adverse dim feeble mizshap struggle afflict disappoint feverish negative suffer alarming disappointment fragile nervousness terrorism apprehension disaster gloom offensive threat apprehensive discomfort gloomy painful tragedy awkward discouragement grim paltry tragic bad dismal harsh pessimistic trouble badly disrupt havoc plague turmoil bitter disruption hit plight unattractive bleak dissatisfied horrible poor undermine bug distort hurt recession undesirable burdensome distortion illegal sank uneasiness corrosive distress insecurity scandal uneasy danger doldrums insidious scare unfavorable daunting downbeat instability sequester unforeseen deadlock emergency interfere sluggish unprofitable deficient erode jeopardize slump unrest depress fail jeopardy sour violent depression failure lack sputter war destruction fake languish stagnant devastation falter loss standstill'

# Tokenize(split) the sentences into words
texts = [[doc] for doc in texts_org.split()]
MyDict_neg = Dictionary(texts)

# express the tf result
tf_result_text = list()
for doc in tqdm.tqdm(tfidf_text[docs_corpus_text]):
    tf_result_text.append([[dictionary_text[id], np.around(freq, decimals=8)] for id, freq in doc])

# # express the tf result
# tf_result_title = list()
# for doc in tqdm.tqdm(tfidf_title[docs_corpus_title]):
#     tf_result_title.append([[dictionary_title[id], np.around(freq, decimals=8)] for id, freq in doc])

i = 1
pn_corpus_neg_text = list()

for doc in tqdm.tqdm(tf_result_text):
    pn_doc = list()
    for [word, freq] in doc:
            if word in MyDict_neg.token2id:
                pn_doc.append([word, freq])
    pn_corpus_neg_text.append(pn_doc)

# pn_corpus_neg_title = list()
# for doc in tqdm.tqdm(tf_result_title):
#     pn_doc = list()
#     for [word, freq] in doc:
#             if word in MyDict_neg.token2id:
#                 pn_doc.append([word, freq])
#     pn_corpus_neg_title.append(pn_doc)

#Construct Positive dictionary

MyDict_pos = "assurance confident exuberant joy prominent Satisfactory unlimited assure constancy facilitate liberal promise Satisfy upbeat attain constructive faith lucrative prompt Sound upgrade attractive cooperate favor manageable proper Soundness uplift auspicious coordinate favorable mediate prosperity Spectacular upside backing credible feasible mend rally Stabilize upward befitting decent fervor mindful readily Stable valid beneficial definitive filial moderation reassure Stable viable beneficiary deserve flatter onward receptive Steadiness victorious benefit desirable flourish opportunity reconcile Steady virtuous benign discern fond optimism refine Stimulate vitality better distinction foster optimistic reinstate Stimulation warm bloom distinguish friendly outrun relaxation Subscribe welcome bolster durability gain outstanding reliable Succeed boom eager generous overcome relief Success boost earnest genuine paramount relieve Successful bountiful ease good particular remarkable Suffice bright easy happy patience remarkably Suit buoyant encourage heal patient repair Support calm encouragement healthy peaceful rescue Supportive celebrate endorse helpful persuasive resolve Surge coherent energetic hope pleasant resolved Surpass comeback engage hopeful please respectable Sweeten comfort enhance hospitable pleased respite Sympathetic comfortable enhancement imperative plentiful restoration Sympathy commend enjoy impetus plenty restore Synthesis compensate enrichment impress positive revival Temperate composure enthusiasm impressive potent revive Thorough concession enthusiastic improve precious ripe Tolerant concur envision improvement pretty rosy tranquil conducive excellent inspire progress salutary tremendous confide exuberance irresistible progressive sanguine undoubtedly"

MyDict_pos =str.lower(MyDict_pos)

# Tokenize(split) the sentences into words
texts = [[doc] for doc in MyDict_pos.split()]

MyDict_pos = Dictionary(texts)


pn_corpus_pos_text = list()

for doc in tqdm.tqdm(tf_result_text):
    pn_doc = list()
    for [word, freq] in doc:
            if word in MyDict_pos.token2id:
                pn_doc.append([word, freq])
    pn_corpus_pos_text.append(pn_doc)


# pn_corpus_pos_title = list()
#
# for doc in tqdm.tqdm(tf_result_title):
#     pn_doc = list()
#     for [word, freq] in doc:
#             if word in MyDict_pos.token2id:
#                 pn_doc.append([word, freq])
#     pn_corpus_pos_title.append(pn_doc)


calc_corpus_text = list()

for i in tqdm.tqdm(range(len(pn_corpus_pos_text))):
    calc_corpus_text.append((pn_corpus_pos_text[i], pn_corpus_neg_text[i]))

# calc_corpus_title = list()
#
# for i in tqdm.tqdm(range(len(pn_corpus_pos_title))):
#     calc_corpus_title.append((pn_corpus_pos_title[i], pn_corpus_neg_title[i]))

calc_corpus_text[0]
len(calc_corpus_text)
DocMetric_text= list()
for doc in calc_corpus_text:
    posfreq_sum = 0
    negfreq_sum = 0
    for pos_word, freq in doc[0]:
        posfreq_sum += freq
        #print(posfreq_sum)
    for neg_word, freq in doc[1]:
        negfreq_sum += freq
            #print(negfreq_sum)
    DocMetric_text.append([posfreq_sum, negfreq_sum])

DocMetric_text[2]
len(DocMetric_text)

# DocMetric_title= list()
# for doc in calc_corpus_title:
#     posfreq_sum = 0
#     negfreq_sum = 0
#     for pos_word, freq in doc[0]:
#         posfreq_sum += freq
#         #print(posfreq_sum)
#     for neg_word, freq in doc[1]:
#         negfreq_sum += freq
#             #print(negfreq_sum)
#     DocMetric_title.append([posfreq_sum, negfreq_sum])

sentiment_index_text = [pos - neg for pos, neg in DocMetric_text]
# sentiment_index_title = [pos - neg for pos, neg in DocMetric_title]
sentiment_index_text[0:5]

#Add sentiment to dataframe
final_df['UK']['sentiment_manual'] = sentiment_index_text

final_df['UK'].tail()
len(sentiment_index_text)
len(final_df['UK'])

################################
#VADER Sentiment analysis
###############################

# positive sentiment: compound score >= 0.05

# neutral sentiment: (compound score > -0.05) and (compound score < 0.05)

# negative sentiment: compound score <= -0.05

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return(score)

vader_sentiment = []
for doc in tqdm.tqdm(final_df['UK']['text'] + final_df['UK']['title']):
    score = sentiment_analyzer_scores(doc)['compound']
    vader_sentiment.append(score)

final_df['UK']['sentiment_vader'] = vader_sentiment




#Standford CoreNLP sentiment
#Connect to Server:  java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000

from pycorenlp import StanfordCoreNLP
nlp_wrapper = StanfordCoreNLP('http://localhost:9000')

final_df['UK'].head()


sentiment_score_standford = []
for doc in tqdm.tqdm(final_df['UK']['text'] + final_df['UK']['title']):
    annot_doc = nlp_wrapper.annotate(doc,
        properties={
           'annotators': 'sentiment',
           'outputFormat': 'json',
           'timeout': 1000,
        })
    sentiment_score_standford.append(annot_doc)


for sentence in annot_doc["sentences"]:
    print ( " ".join([word["word"] for word in sentence["tokens"]]) + " => " \
        + str(sentence["sentimentValue"]) + " = "+ sentence["sentiment"])


#Stanford CoreNLP not allowing to make multiple requests at once


#TextBlob Sentiment analysis

#Reading: Polarity ---> negative vs positive (-1 => +1)

from textblob import TextBlob

sentiment_score_textBlob = []
for doc in tqdm.tqdm(final_df['UK']['text'] + final_df['UK']['title']):
    sentiment_score = TextBlob(doc).sentiment[0]
    sentiment_score_textBlob.append(sentiment_score)

final_df['UK']['sentiment_textblob'] = sentiment_score_textBlob

final_df['UK'].head()


#Aggregate the sentiments

# 1 = positive sentiment
# 0 = neutral
# -1 = negative sentiment



#Calculating the sentiment of manual calculation
final_df['UK']['manual_sentiment'] = [1 if value > 0 else -1 if value < 0 else 0 for value in tqdm.tqdm(final_df['UK']['sentiment_manual'])]

#Calculating the sentiment of vader calculation
final_df['UK']['vader_sentiment'] = [1 if value >= 0.05 else -1 if value <= -0.05 else 0 for value in tqdm.tqdm(final_df['UK']['sentiment_vader'])]

#Calculating the sentiment for textblob calculation
final_df['UK']['textblob_sentiment'] = [1 if value > 0 else -1 if value < 0 else 0 for value in tqdm.tqdm(final_df['UK']['sentiment_textblob'])]

final_df['UK'].to_pickle('news_data/sentiment/sentiment_UK.pkl')
final_list_of_currencies = list(exchange_rate_df.columns)

final_list_of_countries
final_list_of_currencies

#Create Monthly Final dataframe
final_df_monthly = {}
for country in tqdm.tqdm(final_list_of_countries):
    x = final_df[country].resample('MS').sum()
    final_df_monthly.update({country: x})

final_df_monthly['UK']['text'][0]


len(final_df['UK']['text'][0])
x = final_df['UK'].copy()
y = x.resample('MS')
len(y.sum()['text'][0])

from sentiment_analyzer import sa
final_list_of_countries

#Get ride of any observations with 0 for monthly
for country in final_list_of_countries:
    final_df_monthly[country] = final_df_monthly[country][final_df_monthly[country].text != 0]

x = sa(final_list_of_countries[2], load = True)

final_list_of_countries[3:]
final_list_of_countries[2]
#Make an sa instance
final_df_m = final_df
for country in final_list_of_countries[2:]:
    final_df_monthly[country] = sa(country, final_df_monthly[country], False, True, True, True, True, True)

#Daily final df
for country in final_list_of_countries:
    final_df[country] = sa(country, load = True)

#Export data for App
#Sentiment Scores
for country in final_list_of_countries:
    data = final_df[country].final_df[['text', 'title', 'manual_sentiment', 'vader_sentiment', 'textblob_sentiment', 'HIV4_sentiment', 'LM_sentiment', 'aggregate_sentiment']]
    data.to_csv(f'app_data/{country}_data.csv')
final_df['AUS'].final_df.to_csv('app_data/AUS.csv')




final_df['COL']['2012']
# #Aggregate Sentiment
# for country in final_list_of_countries:
#     final_df[country] = final_df_m[country].aggregate_sentiment(country, save = True)
#
#
# #Get Exchange direction column
# for country, currency in zip(final_list_of_countries, final_list_of_currencies):
#     final_df[country] = final_df_m[country].add_exchange_column(exchange_rate_df, country, currency, save = True)



#Calculate Sentiment Score accuracy

vader_score_list = {}
manual_score_list = {}
textblob_score_list = {}
hiv4_score_list = {}
lm_score_list = {}
aggregate_score_list = {}

for country in final_list_of_countries:
    vader_score_list[country] = []
    manual_score_list[country] = []
    textblob_score_list[country] = []
    hiv4_score_list[country] = []
    lm_score_list[country] = []
    aggregate_score_list[country] = []

for country in final_list_of_countries:
    for lag in range(31):
        score = final_df[country].calculate_score(nlag = lag)
        vader_score_list[country].append(score['vader'])
        manual_score_list[country].append(score['manual'])
        textblob_score_list[country].append(score['textblob'])
        hiv4_score_list[country].append(score['hiv4'])
        lm_score_list[country].append(score['lm'])
        aggregate_score_list[country].append(score['aggregate'])

final_list_of_countries
lm_score_list['AUS']
#Export accuracy scores for R app:
accuracy_scores = {}
for country in final_list_of_countries:
    accuracy_scores[country] = {'vader': vader_score_list[country],
                                'manual': manual_score_list[country],
                                'textblob': textblob_score_list[country],
                                'hiv4': hiv4_score_list[country],
                                'lm': lm_score_list[country],
                                'aggregate': aggregate_score_list[country]}

for country in final_list_of_countries:
    accuracy_scores[country] = pd.DataFrame.from_dict(accuracy_scores[country])

for country in final_list_of_countries:
    data = accuracy_scores[country]
    data.to_csv(f'app_data/accuracy_score/{country}_score.csv')


%matplotlib qt
def plot_score(title, score_list):
    # Initialize the figure
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('tab20b')

    # multiple line plot
    num=0
    fig = plt.figure(figsize=(20,18))
    for column in score_list:
        num+=1

        # Find the right spot on the plot
        fig.add_subplot(4,3, num)

        # plot every groups, but discreet
        for v in score_list:
            plt.plot(score_list[v], marker='', color='grey', linewidth=0.6, alpha=0.3)


        # Plot the lineplot
        plt.plot(score_list[column], marker='', color=palette(num), linewidth=2.0, alpha=0.9, label=column)
        plt.locator_params(axis = 'x', nticks=10)
        plt.axhline(50, color="black")

        # Not ticks everywhere
        if num in range(10) :
            plt.tick_params(labelbottom='off')
        if num not in [1,4,7,10] :
            plt.tick_params(labelleft='off')



        # Add title
        plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )

    # general title
    plt.suptitle('', fontsize=16, fontweight=0, color='black', style='italic')

    # Axis title
    plt.text(0.5, 0.02, 'Time', ha='center', va='center')
    plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')
    plt.savefig(title + '.png')

plot_score('vader_score',vader_score_list)
plot_score('manual_score',manual_score_list)
plot_score('textblob_score',textblob_score_list)
plot_score('hiv4_score',hiv4_score_list)
plot_score('lm_score',lm_score_list)
plot_score('aggregate_score',aggregate_score_list)


vader_score_list['AUS']

country_score['AUS']

country_score_lagged_1['AUS']

country_score_lagged_7['AUS']

country_score['JAP']

country_score_lagged_1['JAP']

country_score_lagged_7['JAP']


country_score['EUR']


country_score_lagged_1['EUR']

country_score_lagged_7['EUR']

country_score['CAN']

country_score_lagged_1['CAN']



country_score_lagged_7['CAN']

final_df['AUS'].final_df.head()




def calculate_score(final_df, exchange_rate_df, currency_name):
    #Get UK exchange rates
    exchange = exchange_rate_df[currency_name]
    change = exchange.pct_change()
    change.dropna(inplace = True)
    final_df['exchange_rate'] = change
    binary_column = [1 if value > 0 else -1 for value in final_df['exchange_rate']]
    final_df['exchange_direction'] = binary_column
    final_df[country_name]



#Add percentage change exchange rate to the UK dataframe
final_df['UK']['exchange_rate'] = UK_change

binary_column = [1 if value > 0 else -1 for value in final_df['UK']['exchange_rate']]
final_df['UK']['exchange_direction'] = binary_column

plt.step(final_df['UK'].index, final_df['UK']['manual_sentiment'])
plt.plot(final_df['UK']['manual_sentiment'][:50])
plt.plot(final_df['UK']['vader_sentiment'][:50])
plt.plot(final_df['UK']['textblob_sentiment'][:50])
plt.plot(final_df['UK']['aggregate_sentiment'][:50])
plt.plot(final_df['UK']['exchange_direction'][:50])

plt.legend(['y = manual_sentiment', 'y = vader_sentiment', 'y = texblob', 'y = aggregate'], loc='upper left')
plt.show()

final_df['UK'].head()


final_df['UK'].isna().sum()

final_df['UK'].dropna(inplace = True)

#Check how many times the direction of the exchange rate and the sentiment was the same
vader_score = []
manual_score = []
textblob_score = []
aggregate_score = []
length_of_df = len(final_df['UK'])

#Vader Score
for i in tqdm.tqdm(range(length_of_df)):
    if final_df['UK']['vader_sentiment'].iloc[i] == final_df['UK']['exchange_direction'].iloc[i]:
        vader_score.append(1)

vader_score = (sum(vader_score) / length_of_df) * 100
vader_score

#Manual Score
for i in tqdm.tqdm(range(length_of_df)):
    if final_df['UK']['manual_sentiment'].iloc[i] == final_df['UK']['exchange_direction'].iloc[i]:
        manual_score.append(1)

manual_score = (sum(manual_score) / length_of_df) * 100
manual_score

#TextBlob Score
for i in tqdm.tqdm(range(length_of_df)):
    if final_df['UK']['textblob_sentiment'].iloc[i] == final_df['UK']['exchange_direction'].iloc[i]:
        textblob_score.append(1)

textblob_score = (sum(textblob_score) / length_of_df) * 100
textblob_score

#Aggregate Score
for i in tqdm.tqdm(range(length_of_df)):
    if final_df['UK']['aggregate_sentiment'].iloc[i] == final_df['UK']['exchange_direction'].iloc[i]:
        aggregate_score.append(1)

aggregate_score = (sum(aggregate_score) / length_of_df) * 100
aggregate_score

final_df['UK']['vader_sentiment'].value_counts()

final_df['UK']['manual_sentiment'].value_counts()
final_df['UK']['aggregate_sentiment'].value_counts()

final_df['UK']['textblob_sentiment'].value_counts()

final_df['UK']['exchange_direction'].value_counts()

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
x_train, x_test, y_train, y_test = train_test_split(final_df['UK'][["aggregate_sentiment","vader_sentiment", "manual_sentiment", "textblob_sentiment"]],
                                                    final_df['UK']['exchange_direction'], test_size=0.25, random_state = 42)


logisticRegr.fit(x_train, y_train)

predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)

score

cm = confusion_matrix(y_test, predictions)
print(cm)


final_df['UK'].to_pickle('x.pkl')
##### Setting up the tfidf model #############################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import regularizers
from keras.layers import LSTM
from keras.layers import Embedding
lb = LabelBinarizer(sparse_output = True)    #Probably dont need it as labels are already binary

tfidf = TfidfVectorizer(lowercase = True, max_df = 0.95, stop_words='english')

x = final_df['UK'][['text', 'title', 'aggregate_sentiment']]
y = final_df['UK']['exchange_direction']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 42)

dtm_train = tfidf.fit_transform(x_train['text'] + x_train['title'])
dtm_train

dtm_train.shape[1]

embed_dim = 228
lstm_out = 512
batch_size = 32

model = Sequential()
model.add(Embedding(2372, embed_dim,input_length = 79158, dropout = 0.2))
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

from keras.utils import to_categorical
y_train.shape
early_stop = EarlyStopping(monitor = 'val_err', patience = 5, verbose = 2)
y_train = to_categorical(y_train)
model.fit(dtm_train, y_train, epochs = 500,
          batch_size = 10000, verbose = 2,
          callbacks = [early_stop])



#Creating Neural Network
tag_classifier = Sequential()
#first layer
tag_classifier.add(LSTM(512, activation='relu', kernel_initializer='random_normal', input_dim=79159, kernel_regularizer=regularizers.l2(0.01)))
tag_classifier.add(Dropout(.2))
#second layer
tag_classifier.add(LSTM(64, activation='relu', kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(0.01)))
tag_classifier.add(Dropout(.2))
#output layer
#softmax sums predictions to 1, good for multi-classification
tag_classifier.add(Dense(1, activation ='sigmoid', kernel_initializer='random_normal'))
tag_classifier.summary()

#Compiling
#adam optimizer adjusts learning rate throughout training
#loss function categorical crossentroy for classification
tag_classifier.compile(optimizer ='adam',loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stop = EarlyStopping(monitor = 'val_err', patience = 5, verbose = 2)

tag_classifier.fit(dtm_train, y_train, epochs = 500,
          batch_size = 10000, verbose = 2,
          callbacks = [early_stop])




final_df['UK'].plot.scatter(x = 'aggregate_sentiment', y ='exchange_direction')

sns.violinplot(x='aggregate_sentiment', y = 'exchange_direction', data = final_df['UK'])
sns.violinplot(x='exchange_direction', data = final_df['UK'], col = red)








x = df_EUR.drop_duplicates(keep = 'first')
len(x)
x.head()


x.duplicated().sum()

len(df)
x = consine_similarity_filter(df, doc_text, doc_title, ['brazil'])
df['2008-09-28']
x.head()
len(x)
len(df)
doc_text[0]
x = nlp('Hello my name is America!')
for word in x:
    print([word, word.tag_])

for word in df_nlp['text'][0]:
    print([word, word.pos_, word.has_vector])

print([(token.text, token.vector_norm) for token in doc_text[0]])
str(x)
doc_text[0][1].similarity(doc_text[0][0])
x = nlp('dog cat animal')
x[0].similarity(x[1])
x = nlp('mexico mexican jalisco guadalajara')
x[0].similarity(x[1])
x[3]
for word in doc_text[0]:
    print([word, word.tag_])
for chunk in doc_text[0].noun_chunks:
    print(chunk)

x = nlp("Hello my name is Tawab. I live in America")
for word in x:
    print([word, word.tag_, word.is_oov])
for chunk in x.noun_chunks:
    print(chunk)

for chunk in doc_text[0].noun_chunks:
    print(chunk)
from spacy.lang.en import English
p = English(parser=True, tagger=True, entity=True)






#############################
#Create Word cloud
#############################


from wordcloud import WordCloud

long_string = ','.join(list(df['text'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_image()

final_df['AUS'].head()

#For UK
from wordcloud import WordCloud

long_string = ','.join(list(final_df['CAN']['text'].values + final_df['CAN']['title'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_image()
wordcloud.to_file('CAN_wordCloud.png')




long_string_CHN = ','.join(list(final_df['CHN']['text'].values))
# Create a WordCloud object
wordcloud_CHN = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud_CHN.generate(long_string)

# Visualize the word cloud
wordcloud_CHN.to_image()

wordcloud_CHN.to_file('CHN_wordCloud.png')

long_string_JAP = ','.join(list(final_df['JAP']['text'].values))
# Create a WordCloud object
wordcloud_JAP = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud_JAP.generate(long_string)

# Visualize the word cloud
wordcloud_JAP.to_image()

wordcloud_JAP.to_file('JAP_wordCloud.png')

a = ['canada', 'usa', 'AND']
'and'
y = a.split(' ( ')
y
y[1].split(' | ')
y[0].split('&')

'' in x.iloc[150].text.lower()





x = word_counter(df, ['canada', 'usa'], type = 'or')
sum(x)
len(df)
x = df['text'][0].split("usd, united states")

y = df['text'][0]

'USD' in y.lower()

y.lower()





['stocks', 'recover'] in x[1].lower()
'ins' in x[15].lower()
df.head()

for word in df['text'][0]:
    print(word.text, word.pos_)
df.title
df.iloc[0].text

len(df.text.iloc[2500])
df.title.iloc[2500]
df['2010-12-19']





###################################
#Topic Modeling Code
###################################
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(final_df['UK']['text'] + final_df['UK']['title'])

count_data


lda = LDA(n_components = 5)
lda.fit(count_data)

# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


print_topics = (lda, count_vectorizer, 5)

words = count_vectorizer.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("\nTopic #%d:" % topic_idx)
    print(" ".join([words[i]
                    for i in topic.argsort()[:-8 - 1:-1]]))
###################################
