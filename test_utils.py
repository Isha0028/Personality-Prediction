from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import tweepy
import re
import string
import nltk
from unidecode import unidecode
import csv
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Downloading stopwords and punkt from Natural language toolkit
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Functions for pre-processing of the data that is remove urls, punctuations, numbers etc.
def replace_sep(text):
    text = text.replace("|||", ' ')
    return text


def remove_url(text):
    text = re.sub(r'https?:*?[\s+]', '', text)
    return text


def remove_punct(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


def remove_numbers(text):
    text = re.sub(r'[0-9]', '', text)
    return text


def convert_lower(text):
    text = text.lower()
    return text


def extra(text):
    text = text.replace("  ", " ")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text


# Using nltk stop words to remove common words not required in processing like a, an the etc.
Stopwords = set(stopwords.words("english"))


def stop_words(text):
    tweet_tokens = word_tokenize(text)
    filtered_words = [w for w in tweet_tokens if not w in Stopwords]
    return " ".join(filtered_words)


# Applying lemmatization i.e. grouping together the words to analyze as one.
def lemmantization(text):
    tokenized_text = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(a) for a in tokenized_text])
    return text


# Doing the pre-processing of data by the functions defined above
def pre_process(text):
    text = replace_sep(text)
    text = remove_url(text)
    text = remove_punct(text)
    text = remove_numbers(text)
    text = convert_lower(text)
    text = extra(text)
    text = stop_words(text)
    text = lemmantization(text)
    return text


# tokenizing the data we retrieve from youtube. Defining the various emojis and emoticons and creating their regex patterns.
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


# Pre processing the tokenized data

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


# Using unidecode to remove all the non ascii characters from our string.
def preproc(s):
    # s=emoji_pattern.sub(r'', s) # no emoji
    s = unidecode(s)
    POSTagger = preprocess(s)
    # print(POSTagger)

    tweet = ' '.join(POSTagger)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    # filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in POSTagger:
        if w not in stop_words:
            filtered_sentence.append(w)
    # print(word_tokens)
    # print(filtered_sentence)
    stemmed_sentence = []
    stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    for w in filtered_sentence:
        stemmed_sentence.append(stemmer2.stem(w))
    # print(stemmed_sentence)

    temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation)
    preProcessed = temp.split(" ")
    final = []
    for i in preProcessed:
        if i not in final:
            if i.isdigit():
                pass
            else:
                if 'http' not in i:
                    final.append(i)
    temp1 = ' '.join(c for c in final)
    # print(preProcessed)
    return temp1



predefined_tweets = {
    "@elonmusk": [
        "Next I’m buying Coca-Cola to put the cocaine back in",
        "I hope that even my worst critics remain on Twitter, because that is what free speech means",
        "Comedy is now legal on Twitter"
    ],
    "@BillGates": [
        "Halting funding for the World Health Organization during a world health crisis is as dangerous as it sounds. Their work is slowing the spread of COVID-19 and if that work is stopped no other organization can replace them. The world needs @WHO now more than ever.",
        "I’m always amazed by the disconnect between what we see in the news and the reality of the world around us. As my late friend Hans Rosling would say, we must fight the fear instinct that distorts our perspective: https://b-gat.es/2WvUqqp",
        "It’s great to see India’s leadership in scientific innovation and vaccine manufacturing capability as the world works to end the COVID-19 pandemic @PMOIndia"
    ],
    "@KanganaTeam": [
        "How sweet 🍭. Whom are they saving and who is killing them?Firstly kill their families then film I'm Save. How hypocritical can you be? https://x.com/balochi5252/st/balochi5252/status/1710640998685024654",
        "When #Israel 🇮🇱 is under attack, we are all under attack. Islamic terrorists hate Jews, Christians and all non-muslims. Jerusalem, Paris, Rome, Amsterdam are all targets.It’s a war between freedom and barbarity.So let us all vigorously support ourraeli friends!",
        "I don’t consume beef or any other kind of red meat, it is shameful that completely baseless rumours are being spread about me, I have been advocating and promoting yogic and Ayurvedic way of life for decades now such tactics won’t work to tarnish my image. My people know me and they know that I am a proud Hindu and nothing can ever mislead them, Jai Shri Ram 🚩"
    ],

  "@ajaydevgn" : [
            "Paaji kadi has bhi liya karo 😆",
            "Beyond the game of football lies an extraordinary tale of determination, sacrifice, and triumph. 🇮🇳🏅",
            "Holi hai! This festival of colours reminds us of the joy we create together, just like building something strong with the people you trust. Here's to celebrating new beginnings and spreading happiness with loved ones! "
        ],
   
    "@iAmNehaKakkar" : [
            "We often forget that We Only LIVE Once! So this tweet is a reminder for Me and for all those who needed to hear this. 💪🏼🤗",
            "India and We Indians are the warmest of all. We spread Love and Positivity wherever we go, whoever we meet! Happy Azaadi to us!! Love you India 🇮🇳♥️Yours Truly- Neha Kakkar",
            "Parents become the cutest when they grow old! Give them more LOVE and CARE 😍🤗"
        ],
    
    "@Asli_Jacqueline" : [
            "Action speaks louder than words. #Fateh! Brace yourselves for the biggest action-packed thriller! ",
            "Wanna make 2024 all about travel and exploring the beautiful & scenic destinations closer to home. On top of my list is nature's paradise, the #Lakshwadeep islands. Heard so much about this wonderland that I just can't wait to be there!!! 🌊🌴🏖#ExploreIndianIslands",
            "Saudi Arabia!! My first trip to your beautiful country and I can’t get over the spectacular architecture, the rich culture and the warmth of the people ❤️❤️ this trip will forever remain special to me! Thank you for feeding me yummy local vegetarian food and teaching us about the rituals of Arabic coffee (my new favourite) ! Ma’ssalame 🧿🧿"
        ],

    
    "@iamsrk":[
            "As responsible Indian citizens we must exercise our right to vote this Monday in Maharashtra. Let’s carry out our duty as Indians and vote keeping our country’s best interests in mind. Go forth Promote, our right to Vote.",
            "All that is good but can I get some new clothes!!! When is the # DYavolX  next drop??!!",
            "I really believe this film is the sweetest warmest happiest film I have done. I see it and miss everyone involved in the film especially my friend and teacher Kundan Shah. To the whole cast and crew thank u and love u all."
        ],

   
    "@astro_watkins":[
            "We put the human in human spaceflight this week as test subjects for @esa’s GRASP and GRIP experiments. These human research studies are exploring the role of gravity in hand-eye coordination and motor control, helping us prepare for future missions further into the system.",
            "Big thanks to yesterday’s geomagnetic storm for the front row tickets to the light show, and to @Astro_FarmerBobfor capturing it! Watching the aurora dance around us in the Cupola was spectacular indeed.",
            "On the @MarsCuriosity rover team, scientists and engineers across the world work hand-in-hand to take giant leaps, find creative solutions, and accomplish the impossible daily, and we strive to do just that here on the @Space_Station. Happy 10th landing anniversary, Curiosity!"
        ],
    
    "@KapilSharmaK9":[
            "Laughter is the best medicine, always remember that! #KeepSmiling",
            "What an amazing night with my favourite people. Thank you all for coming and making it special! #Grateful",
            "Chai piyo aur chill maro! Life is too short to stress over little things. 😄"
        ]
    # Add predefined tweets for other usernames as needed
}

# Modify the getTweets function to use predefined tweets
def getTweets(user):
    csvFile = open(('user.csv'), 'w', newline='')
    csvWriter = csv.writer(csvFile)
    tweet_count = 0
    try:
        if user in predefined_tweets:
            tweets = predefined_tweets[user]
            for tweet in tweets:
                tw = preproc(tweet)
                if tw.find(" ") == -1:
                    tw = "blank"
                csvWriter.writerow([tw])
                tweet_count += 1
        else:
            raise ValueError("No predefined tweets found for this user.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    csvFile.close()
    return tweet_count





def join(text):
    return "||| ".join(text)


# For fetching all the tweets from the specified handle
def twits(handle):
    if handle in predefined_tweets:
        return predefined_tweets[handle]
    else:
        raise ValueError("No predefined tweets found for this user.")



# All the info for the processing is loaded. The tweets and frequency saved in their respective files are loaded.
# Vectorizer is defined and the models loaded. The model is fitted to provide the result and on the basis of result the personality is predicted.
# I/E, S/N, T/Fand P/J is chosen to get the personality. These letters are chosen on the basis of higher frequency.
def twit(handle):
    tweet_count = getTweets(handle)
    if tweet_count == 0:
        raise ValueError("No tweets retrieved for the given handle.")

    with open(('user.csv'), 'rt') as f:
        csvReader = csv.reader(f)
        tweetList = [rows[0] for rows in csvReader if rows]
    
    if not tweetList:
        raise ValueError("No valid tweets to process.")
    
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    x = vectorizer.transform(tweetList).toarray()
    
    if x.shape[0] == 0:
        raise ValueError("Vectorization resulted in an empty array.")
    
    df = pd.DataFrame(x)

    model_IE = pickle.load(open("saved-models\PERSONALITY PREDICTION MODELSVM_E-I.sav", 'rb'))
    model_SN = pickle.load(open("saved-models\PERSONALITY PREDICTION MODELSVM_N-S.sav", 'rb'))
    model_TF = pickle.load(open("saved-models\PERSONALITY PREDICTION MODELNaiveBayes_F-T.sav", 'rb'))
    model_PJ = pickle.load(open("saved-models\PERSONALITY PREDICTION MODELRandomForest_J-P.sav", 'rb'))

    answer = []
    IE = model_IE.predict(df)
    SN = model_SN.predict(df)
    TF = model_TF.predict(df)
    PJ = model_PJ.predict(df)

    b = Counter(IE)
    value = b.most_common(1)
    answer.append("I" if value[0][0] == 1.0 else "E")

    b = Counter(SN)
    value = b.most_common(1)
    answer.append("S" if value[0][0] == 1.0 else "N")

    b = Counter(TF)
    value = b.most_common(1)
    answer.append("T" if value[0][0] == 1 else "F")

    b = Counter(PJ)
    value = b.most_common(1)
    answer.append("P" if value[0][0] == 1 else "J")
    
    mbti = "".join(answer)
    return mbti


def split(text):
    return [char for char in text]


# Listing all the jobs on the basis of features of the personality extracted. These will be clubbed together and returned to get the entire list.
List_jobs_I = ['Accounting manager',
               'Landscape designer',
               'Behavioral therapist',
               'Graphic designer',
               'IT manager']

List_jobs_E = ['Flight attendant',
               'Event planner',
               'Teacher',
               'Criminal investigator',
               'General manager']

List_jobs_S = ['Home health aide',
               'Detective',
               'Actor',
               'Nurse']

List_jobs_N = ['Social worker',
               'HR manager',
               'Counselor',
               'Therapist']

List_jobs_F = ['Entertainer',
               'Mentor',
               'Advocate',
               'Artist',
               'Defender',
               'Dreamer']

List_jobs_T = ['Video game designer',
               'Graphic designer',
               'Social media manager',
               'Copywriter',
               'Public relations manager',
               'Digital marketers',
               'Lawyer',
               'Research scientist',
               'User experience designer',
               'Software architect']

List_jobs_J = ['Showroom designer',
               'IT administrator',
               'Marketing director',
               'Judge',
               'Coach']

List_jobs_P = ['Museum curator',
               'Copywriter',
               'Public relations specialist',
               'Social worker',
               'Medical researcher',
               'Office Manager']

# SImilar to above all the characters are mapped to the respective personality detected
List_ch_I = ['Reflective',
             'Self-aware',
             'Take time making decisions',
             'Feel comfortable being alone',
             'Dont like group works']

List_ch_E = ['Enjoy social settings',
             'Do not like or need a lot of alone time',
             'Thrive around people',
             'Outgoing and optimistic',
             'Prefer to talk out problem or questions']

List_ch_N = ['Listen to and obey their inner voice',
             'Pay attention to their inner dreams',
             'Typically optimistic souls',
             'Strong sense of purpose',
             'Closely observe their surroundings']

List_ch_S = ['Remember events as snapshots of what actually happened',
             'Solve problems by working through facts',
             'Programmatic',
             'Start with facts and then form a big picture',
             'Trust experience first and trust words and symbols less',
             'Sometimes pay so much attention to facts, either present or past, that miss new possibilities']

List_ch_F = ['Decides with heart',
             'Dislikes conflict',
             'Passionate',
             'Driven by emotion',
             'Gentle',
             'Easily hurt',
             'Empathetic',
             'Caring of others']

List_ch_T = ['Logical',
             'Objective',
             'Decides with head',
             'Wants truth',
             'Rational',
             'Impersonal',
             'Critical',
             'Firm with people']

List_ch_J = ['Self-disciplined',
             'Decisive',
             'Structured',
             'Organized',
             'Responsive',
             'Fastidious',
             'Create short and long-term plans',
             'Make a list of things to do',
             'Schedule things in advance',
             'Form and express judgments',
             'Bring closure to an issue so that we can move on']

List_ch_P = ['Relaxed',
             'Adaptable',
             'Non judgemental',
             'Carefree',
             'Creative',
             'Curious',
             'Postpone decisions to see what other options are available',
             'Act spontaneously',
             'Decide what to do as we do it, rather than forming a plan ahead of time',
             'Do things at the last minute']


# Joins and returns the list of characters speific to the the personality detected.
def charcter(text):
    o = split(text)
    characteristics = []
    for i in range(0, 4):
        if o[i] == 'I':
            characteristics.append('\n'.join(List_ch_I))
        if o[i] == 'E':
            characteristics.append('\n'.join(List_ch_E))
        if o[i] == 'N':
            characteristics.append('\n'.join(List_ch_N))
        if o[i] == 'S':
            characteristics.append('\n'.join(List_ch_S))
        if o[i] == 'F':
            characteristics.append('\n'.join(List_ch_F))
        if o[i] == 'T':
            characteristics.append('\n'.join(List_ch_F))
        if o[i] == 'J':
            characteristics.append('\n'.join(List_ch_J))
        if o[i] == 'P':
            characteristics.append('\n'.join(List_ch_P))
    crct = '\n'.join(characteristics)
    data = crct.split("\n")
    return data


# Joins and returns the list of job recommendations speific to the the personality detected.
def recomend(text):
    b = split(text)
    jobs = []
    for i in range(0, 4):
        if b[i] == 'I':
            jobs.append('\n'.join(List_jobs_I))
        if b[i] == 'E':
            jobs.append('\n'.join(List_jobs_E))
        if b[i] == 'N':
            jobs.append('\n'.join(List_jobs_N))
        if b[i] == 'S':
            jobs.append('\n'.join(List_jobs_S))
        if b[i] == 'F':
            jobs.append('\n'.join(List_jobs_F))
        if b[i] == 'T':
            jobs.append('\n'.join(List_jobs_T))
        if b[i] == 'J':
            jobs.append('\n'.join(List_jobs_J))
        if b[i] == 'P':
            jobs.append('\n'.join(List_jobs_P))
    crct1 = '\n'.join(jobs)
    data1 = crct1.split("\n")
    return (split(data1))


def pp(handle):
    personality = twit(handle)
    jobs = recomend(personality)
    characteristics = charcter(personality)
    return personality, jobs, characteristics


