from imports import *

class sa:
    def __init__(self, country_name, df = None, load = False, manual = False, vader = False, textblob = False, HIV4 = False, LM = False):
        self.df = df
        self.final_df = self.calculate_sentiment(country_name, load, manual, vader, textblob, HIV4, LM)

    def calculate_sentiment(self, country_name, load, manual, vader, textblob, HIV4, LM):
        if load:
            result = pd.read_pickle('news_data/sentiment/sentiment_' + country_name + '.pkl')
        else:
            result = self.df.copy()
        if manual:
            manual_sentiment = self.manual_sentiment()
            result['manual_sentiment_score'] = manual_sentiment
            #Calculating the sentiment of manual calculation
            result['manual_sentiment'] = [1 if value > 0 else -1 if value < 0 else 0 for value in tqdm.tqdm(result['manual_sentiment_score'])]

        if vader:
            vader_sentiment = self.vader_sentiment()
            result['vader_sentiment_score'] = vader_sentiment
            #Calculating the sentiment of vader calculation
            result['vader_sentiment'] = [1 if value >= 0.05 else -1 if value <= -0.05 else 0 for value in tqdm.tqdm(result['vader_sentiment_score'])]

        if textblob:
            textblob_sentiment = self.textblob_sentiment()
            result['textblob_sentiment_score'] = textblob_sentiment
            #Calculating the sentiment for textblob calculation
            result['textblob_sentiment'] = [1 if value > 0 else -1 if value < 0 else 0 for value in tqdm.tqdm(result['textblob_sentiment_score'])]

        if HIV4:
            HIV4_sentiment = self.HIV4_sentiment()
            result['HIV4_sentiment_score'] = HIV4_sentiment
            result['HIV4_sentiment'] = [1 if value > 0 else -1 if value < 0 else 0 for value in tqdm.tqdm(result['HIV4_sentiment_score'])]

        if LM:
            LM_sentiment = self.LM_sentiment()
            result['LM_sentiment_score'] = LM_sentiment
            result['LM_sentiment'] = [1 if value > 0 else -1 if value < 0 else 0 for value in tqdm.tqdm(result['LM_sentiment_score'])]


        if manual or vader or textblob or HIV4 or LM:
            result.to_pickle('news_data/sentiment/sentiment_' + country_name + '.pkl')

        return result

    def manual_sentiment(self):
        #Vectorize data and add title and text
        doc_text = []
        for text in tqdm.tqdm(self.df['text'] + self.df['title']):
            doc_text.append(nlp(text))

        # Clean the text
        token_tot_text = list()
        for i in tqdm.tqdm(range(len(doc_text))):
            tokens_i = [token.text.lower() for token in doc_text[i] if not token.is_stop]
            token_tot_text.append(tokens_i)

        # Dictionary of the whole corpus
        #dataset = [d.split() for d in token_tot]
        dictionary_text = Dictionary(token_tot_text)

        #The corpus
        docs_corpus_text = [dictionary_text.doc2bow(doc) for doc in token_tot_text]

        # Create the TF-IDF model
        tfidf_text = models.TfidfModel(docs_corpus_text, smartirs='ntc')

        #Construct negative dictionary

        texts_org = ' adverse dim feeble mizshap struggle afflict disappoint feverish negative suffer alarming disappointment fragile nervousness terrorism apprehension disaster gloom offensive threat apprehensive discomfort gloomy painful tragedy awkward discouragement grim paltry tragic bad dismal harsh pessimistic trouble badly disrupt havoc plague turmoil bitter disruption hit plight unattractive bleak dissatisfied horrible poor undermine bug distort hurt recession undesirable burdensome distortion illegal sank uneasiness corrosive distress insecurity scandal uneasy danger doldrums insidious scare unfavorable daunting downbeat instability sequester unforeseen deadlock emergency interfere sluggish unprofitable deficient erode jeopardize slump unrest depress fail jeopardy sour violent depression failure lack sputter war destruction fake languish stagnant devastation falter loss standstill'

        # Tokenize(split) the sentences into words
        texts = [[doc] for doc in texts_org.split()]
        MyDict_neg = Dictionary(texts)

        # express the tf result
        tf_result_text = list()
        for doc in tqdm.tqdm(tfidf_text[docs_corpus_text]):
            tf_result_text.append([[dictionary_text[id], np.around(freq, decimals=8)] for id, freq in doc])

        pn_corpus_neg_text = list()

        for doc in tqdm.tqdm(tf_result_text):
            pn_doc = list()
            for [word, freq] in doc:
                    if word in MyDict_neg.token2id:
                        pn_doc.append([word, freq])
            pn_corpus_neg_text.append(pn_doc)

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

        calc_corpus_text = list()

        for i in tqdm.tqdm(range(len(pn_corpus_pos_text))):
            calc_corpus_text.append((pn_corpus_pos_text[i], pn_corpus_neg_text[i]))

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

        sentiment_index_text = [pos - neg for pos, neg in DocMetric_text]

        return sentiment_index_text


    def sentiment_analyzer_scores(self, sentence):
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(sentence)
        return(score)


    def vader_sentiment(self):
        ###############################

        # positive sentiment: compound score >= 0.05

        # neutral sentiment: (compound score > -0.05) and (compound score < 0.05)

        # negative sentiment: compound score <= -0.05
        vader_sentiment = []
        for doc in tqdm.tqdm(self.df['text'] + self.df['title']):
            score = self.sentiment_analyzer_scores(doc)['compound']
            vader_sentiment.append(score)

        return vader_sentiment



    def textblob_sentiment(self):
        #Reading: Polarity ---> negative vs positive (-1 => +1)

        sentiment_score_textBlob = []
        for doc in tqdm.tqdm(self.df['text'] + self.df['title']):
            sentiment_score = TextBlob(doc).sentiment[0]
            sentiment_score_textBlob.append(sentiment_score)

        return sentiment_score_textBlob


    def HIV4_sentiment(self):
        #Harvard IV-4 Dictionary
        hiv4 = ps.HIV4()
        score_list = []
        for text in tqdm.tqdm(self.df['text'] + self.df['title']):
            tokens = hiv4.tokenize(text)
            score = hiv4.get_score(tokens)
            score_list.append(score['Polarity'])

        return score_list

    def LM_sentiment(self):
        #Loughran and McDonald dictionary

        lm = ps.LM()
        score_list = []
        for text in tqdm.tqdm(self.df['text'] + self.df['title']):
            tokens = lm.tokenize(text)
            score = lm.get_score(tokens)
            score_list.append(score['Polarity'])

        return score_list

    def aggregate_sentiment(self, country_name, manual = True, vader = True, textblob = True, HIV4 = True, LM = True, save = False):
        result = self.final_df.copy()
        sentiment_list = []
        if manual:
            sentiment_list.append('manual_sentiment')
        if vader:
            sentiment_list.append('vader_sentiment')
        if textblob:
            sentiment_list.append('textblob_sentiment')
        if HIV4:
            sentiment_list.append('HIV4_sentiment')
        if LM:
            sentiment_list.append('LM_sentiment')

        result['aggregate_sentiment'] = 0
        for sentiment_name in sentiment_list:
            result['aggregate_sentiment'] = result['aggregate_sentiment'] + result[sentiment_name]

        result['aggregate_sentiment'][result['aggregate_sentiment'] > 1] = 1
        result['aggregate_sentiment'][result['aggregate_sentiment'] < -1] = -1

        if save:
            result.to_pickle('news_data/sentiment/sentiment_' + country_name + '.pkl')

        return result

    def add_exchange_column(self, exchange_rate_df, country_name, currency_name, save = False):
        result = self.final_df.copy()
        exchange = exchange_rate_df[currency_name]
        change = exchange.pct_change()
        change.dropna(inplace = True)
        result['exchange_rate'] = change
        binary_column = [1 if value > 0 else -1 for value in result['exchange_rate']]
        result['exchange_direction'] = binary_column
        result.dropna(inplace = True)
        if save:
            result.to_pickle('news_data/sentiment/sentiment_' + country_name + '.pkl')

        return result

    def calculate_score(self, nlag = 0, manual = True, vader = True, textblob = True, HIV4 = True, LM = True, aggregate = True):
        result = self.final_df.copy()
        length_of_df = len(result)
        score_result = {}

        if vader:
            vader_score = []
            #Vader Score
            for i in tqdm.tqdm(range(length_of_df - nlag)):
                if result['vader_sentiment'].iloc[i] == result['exchange_direction'].iloc[i + nlag]:
                    vader_score.append(1)

            vader_score = (sum(vader_score) / length_of_df) * 100
            score_result.update({'vader': vader_score})

        if manual:
            manual_score = []
            #Manual Score
            for i in tqdm.tqdm(range(length_of_df - nlag)):
                if result['manual_sentiment'].iloc[i] == result['exchange_direction'].iloc[i + nlag]:
                    manual_score.append(1)

            manual_score = (sum(manual_score) / length_of_df) * 100
            score_result.update({'manual': manual_score})

        if textblob:
            textblob_score = []
            #TextBlob Score
            for i in tqdm.tqdm(range(length_of_df - nlag)):
                if result['textblob_sentiment'].iloc[i] == result['exchange_direction'].iloc[i + nlag]:
                    textblob_score.append(1)

            textblob_score = (sum(textblob_score) / length_of_df) * 100
            score_result.update({'textblob': textblob_score})

        if HIV4:
            hiv4_score = []
            #HIV4 Score
            for i in tqdm.tqdm(range(length_of_df - nlag)):
                if result['HIV4_sentiment'].iloc[i] == result['exchange_direction'].iloc[i + nlag]:
                    hiv4_score.append(1)

            hiv4_score = (sum(hiv4_score) / length_of_df) * 100
            score_result.update({'hiv4': hiv4_score})

        if LM:
            lm_score = []
            #LM Score
            for i in tqdm.tqdm(range(length_of_df - nlag)):
                if result['LM_sentiment'].iloc[i] == result['exchange_direction'].iloc[i + nlag]:
                    lm_score.append(1)

            lm_score = (sum(lm_score) / length_of_df) * 100
            score_result.update({'lm': lm_score})

        if aggregate:
            aggregate_score = []
            #Aggregate Score
            for i in tqdm.tqdm(range(length_of_df - nlag)):
                if result['aggregate_sentiment'].iloc[i] == result['exchange_direction'].iloc[i + nlag]:
                    aggregate_score.append(1)

            aggregate_score = (sum(aggregate_score) / length_of_df) * 100
            score_result.update({'aggregate': aggregate_score})

        return score_result
