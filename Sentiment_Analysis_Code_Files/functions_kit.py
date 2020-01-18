from imports import *


def remove_HTMLtags(text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)

#Count number of news per day and visualize it:
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
        plt.title("Daily FOREX News Release Frequency on Investing.com")
        plt.ylabel('Number of News Releases')
        plt.xlabel('Date')
        plt.bar(news_release_rate.index, news_release_rate.number_of_news_releases, color = 'r')
    return news_release_rate

#Count a Specific word list in text
def word_counter(df, list_, type = 'or'):
    result = [0] * len(df.text)
    for i, text in zip(range(len(df.text)),df.text):
        if type == 'or':
            for word in list_:
                if word in text.lower():
                    result[i] = 1
                    break
        else:
            and_counter = 0
            for word in list_:
                if word in text.lower():
                    and_counter = 1
                else:
                    and_counter = 0
                    break
            if and_counter == 1:
                result[i] = 1
    return result


#Basic Text Cleaning
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
            string = string.replace(x, "")

    # Print string without punctuation
    return string

def regex_cleanUp(text_doc):

        result = []
        for text in text_doc:
            x = remove_HTMLtags(text)
            result.append(x)
        text_doc = result

        result = []
        for text in tqdm.tqdm(text_doc):
            x = Punctuation(text)
            result.append(x)
        text_doc = result

        return text_doc
############################
