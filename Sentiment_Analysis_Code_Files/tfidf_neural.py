from imports import *
from functions_kit import *


class tn:
    def __init__(self, df):
        self.df = df


    def prepare_data(self, exchange_rate_df, currency_name):

        exchange = exchange_rate_df[currency_name]
        change = exchange.pct_change()
        change.dropna(inplace = True)

        model_data = pd.concat([self.df, change], axis = 1)
        model_data.dropna(inplace = True)

        #Get positive and negative change binary column:
        binary_column = [1 if value > 0 else 0 for value in model_data[currency_name]]
        model_data['change'] = binary_column

        x = model_data[['text', 'title']]
        y = model_data['change']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 42)

        return x_train, x_test, y_train, y_test




    def setup_neural_network(self, x_train, x_test, y_train, y_test, currency_name, regularizer_value = 0.01, dropout_value = 0.2, load = False):

        if load:
            tag_classifier = load_model('news_data/' + currency_name + '_neural_model.h5')
            return tag_classifier
        else:

            dtm_train = tfidf.fit_transform(x_train['text'] + x_train['title'])

            #Creating Neural Network
            tag_classifier = Sequential()
            #first layer
            tag_classifier.add(Dense(512, activation='relu', kernel_initializer='random_normal', input_dim=dtm_train.shape[1], kernel_regularizer=regularizers.l2(regularizer_value)))
            tag_classifier.add(Dropout(dropout_value))
            #second layer
            tag_classifier.add(Dense(64, activation='relu', kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(regularizer_value)))
            tag_classifier.add(Dropout(dropout_value))
            #output layer
            #softmax sums predictions to 1, good for multi-classification
            tag_classifier.add(Dense(1, activation ='sigmoid', kernel_initializer='random_normal'))
            tag_classifier.summary()

            #Compiling
            #adam optimizer adjusts learning rate throughout training
            #loss function categorical crossentroy for classification
            tag_classifier.compile(optimizer ='adam',loss = 'binary_crossentropy', metrics = ['accuracy'])
            early_stop = EarlyStopping(monitor = 'loss', patience = 1, verbose = 2)

            tag_classifier.fit(dtm_train, y_train, epochs = 200,
                      batch_size = 500, verbose = 2,
                      callbacks = [early_stop])

            if load == False:
                tag_classifier.save('news_data/' + currency_name + '_neural_model.h5')

            return tag_classifier


    def get_predicted_value(self, tag_classifier, test = False, x_train = None, x_test = None, y_train = None, y_test = None):
        if test:
            #Get predictions on test data
            dtm_test = tfidf.transform(x_test['text'] + x_test['title'])
            y_pred_test = tag_classifier.predict(dtm_test)
            y_predictions = [1 if value[0] >= 0.5 else 0 for value in y_pred_test]
            return y_predictions

        else:
            dtm_train = tfidf.transform(x_train['text'] + x_train['title'])
            y_pred = tag_classifier.predict(dtm_train)
            y_predictions = [1 if value[0] >= 0.5 else 0 for value in y_pred]
            return y_predictions
