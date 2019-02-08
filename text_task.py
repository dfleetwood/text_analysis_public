#TODO: Sort out imports

from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_multiple_whitespaces, strip_numeric
import re
import pickle
#from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import HTML as html_print
from gensim.models import Word2Vec, FastText
from fastai.text import *

class TextTask:

    def __init__(self):
        self.texts = {}
        self.tfidf_models = {}
        self.texts_tok = {}
        self.gensim_dicts = {}
        self.tf_idfs = {}
        self.emb_models = {}
        self.embeds = {}
        self.sent_embeds = {}

    def add_text(self, text, text_tag):
        self.texts.update ({text_tag: text})

    def clean_text (self, text_tag, processes = ["urls", "punctuation", "numeric", "lower"]):
        text = self.texts [text_tag]

        #print (text)

        if "urls" in processes:
            text = [re.sub(r"(?:\@|https?\://)\S+", "", str(x)) for x in text]
            text = [re.sub(r' +', ' ', str(x)) for x in text]
        if "stopwords" in processes:
            text = [remove_stopwords (x) for x in text]
        if "punctuation" in processes:
            text = [strip_punctuation(x) for x in text]
        if "numeric" in processes:
            text = [strip_numeric(x) for x in text]

        text = [x.replace('"', "") for x in text]
        text = [x.replace('©', "") for x in text]
        text = [x.replace('\n', " ") for x in text]
        text = [x.replace('\r', ".") for x in text]
        text = [x.replace('QT', " ") for x in text]
        text = [x.replace('RT', " ") for x in text]
        text = [x.replace('#', " ") for x in text]
        text = [strip_multiple_whitespaces(x) for x in text]
        text = [x.strip() for x in text]

        if "lower" in processes:
            text = [x.lower() for x in text]
        # clean_text = [nltk.sent_tokenize (x) for x in  clean_text]

        self.texts[text_tag] = text

    def get_text_tags(self):
        return (self.texts.keys())

    def save (self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def tokenize (self, text_tag):
        #TODO tokenise properly (using spacy or other tokeniser)
        text_tok = self._tokenize (self.texts [text_tag])
        #text_tok = [x.split (" ") for x in self.texts [text_tag]]
        self.texts_tok.update ({text_tag: text_tok})

    def _tokenize (self, text):
        #TODO tokenise properly (using spacy or other tokeniser)
        text_tok = [x.split (" ") for x in text]
        return (text_tok)

    def fit_tf_idf (self, text_tag, max_vocab = 30000):
        text = self.texts [text_tag]

        tfidf_model = TfidfVectorizer(lowercase = False, max_features = max_vocab)
        tf_idf_vec = tfidf_model.fit_transform(text)

        self.tfidf_models.update({text_tag: tfidf_model})
        self.tf_idfs.update({text_tag: tf_idf_vec})

    def _embed_line (self, text_tok, embedding_model, emb_length):
        line_embedding = []
        for token in text_tok:
            try:
                line_embedding.append (embedding_model [token])
            except KeyError as e:
                #Not in vocabulary
                #print (e)
                line_embedding.append([0.]*emb_length)
        return (line_embedding)

    #TODO: Topic modelling

    def load_embedding_model (self, pretrained_model_tag):
        # TODO: Loading the model takes an age, is there a way to speed it up? E.g. load model directly?
        if not (pretrained_model_tag in self.emb_models.keys()):
            print("Loading model")
            emb_model = api.load(pretrained_model_tag)
            self.emb_models.update({pretrained_model_tag: emb_model})


    def embed_text (self, text_tags, pretrained_model_tag):
       #TODO: reloading the model every time will be slow, but don't save in the class, it'll be huge

       #if not (pretrained_model_tag in self.emb_models.keys()):
       self.load_embedding_model(pretrained_model_tag)
           #emb_model = api.load(pretrained_model_tag)
           #self.emb_models.update ({pretrained_model_tag: emb_model})

       emb_model = self.emb_models [pretrained_model_tag]

       if ~(text_tags in self.texts_tok.keys()):
           self.tokenize (text_tags)

       text_tok = self.texts_tok [text_tags]
       print ("Embedding " + str (len (text_tok)) + " texts")

       emb_length = emb_model['a'].shape[0]

       embeds = []
       for i in range (len (text_tok)):
           if (i % 1000) == 0:
               print (i)
           embeds.append (self._embed_line (text_tok[i], emb_model, emb_length))

       self.embeds.update ({text_tags: {pretrained_model_tag: np.array (embeds)}})


    def get_available_embeddings(self):
        return (list(api.info()["models"].keys()))


    def embedding_lookup (self, text_tag, pretrained_model_tag, lookup_text, lookup_agg_method = np.mean, text_agg_method = np.max):
        #TODO: Convert to pytorch so can use GPU if one's available

        #if not (pretrained_model_tag in self.emb_models.keys()):
        self.load_embedding_model(pretrained_model_tag)

        emb_model = self.emb_models[pretrained_model_tag]

        lookup_text_tok = self._tokenize([lookup_text])
        # print (lookup_text_tok)
        emb_length = emb_model['a'].shape[0]
        lookup_text_emb = self._embed_line(lookup_text_tok, emb_model, emb_length)
        lookup_text_emb = np.array(lookup_text_emb)
        lookup_text_emb = lookup_text_emb.squeeze(0)

        embs = self.embeds[text_tag][pretrained_model_tag]

        emb_sims = []
        emb_sims_max = []
        for i in range(0, len(embs)):
            if (i % 1000) == 0:
                print(i)

            lin_cos_sim = cosine_similarity(embs[i], lookup_text_emb)
            lin_cos_sim = lookup_agg_method(lin_cos_sim, axis=1)
            emb_sims.append(lin_cos_sim)
            max_sims = text_agg_method(lin_cos_sim)
            emb_sims_max.append(max_sims)

        emb_sims = np.array(emb_sims)
        emb_sims_max = np.array(emb_sims_max)

        ordering = np.argsort(-emb_sims_max)

        return (emb_sims, emb_sims_max, ordering)


    def word_to_sentence_embedding (self, text_tag, pretrained_model_tag, stopwords = None):
        text_embeds = self.embeds[text_tag][pretrained_model_tag]
        text_tok = self.texts_tok[text_tag]

        if stopwords is None:
            stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                         "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
                         'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
                         'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                         'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                         'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                         'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                         'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                         'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                         'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                         'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                         'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
                         'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
                         'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
                         'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                         "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                         "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                         'wouldn', "wouldn't"]

        sent_emb = []

        print ("Embedding " + str (len(text_tok)) + " sentences")
        for i in range(len(text_tok)):
            if (i % 1000) == 0:
                print(i)
            assert (len(text_embeds[i]) == len(text_tok[i]))
            sent_emb.append(
                np.average(np.array(text_embeds[i])[np.where(~np.isin(text_tok[i], stopwords))[0]], axis=0))

        sent_emb = np.array(sent_emb)

        self.sent_embeds.update ({text_tag: {pretrained_model_tag: sent_emb}})

    def train_embed_model(self, text_tag, model_tag, model_type = ["w2vec", "ft"], embed_size = 200):
        assert (model_type in ["w2vec", "ft"])

        if not (text_tag in self.texts_tok.keys()):
            self.tokenize(text_tag)

        print ("Training model")
        if model_type == "w2vec":
            model = Word2Vec(self.texts_tok[text_tag], size=embed_size, window=5, min_count=1, workers=4)

        if model_type == "ft":
            model = FastText(self.texts_tok[text_tag], size=embed_size, window=5, min_count=1, workers=4)

        self.emb_models.update ({model_tag: model})


    def colour_attention (self, text_tag, attention_weights, subset = None):
        from sklearn.utils.extmath import softmax

        if not (text_tag in self.texts_tok.keys()):
            self.tokenize(text_tag)
        text_tok = self.texts_tok[text_tag]

        if subset is not None:
            text_tok = text_tok[subset]
            attention_weights = attention_weights[subset]

        def cstr(s, color='black'):
            return "<text style=background-color:{}>{}</text>".format(color, s)

        def norm(x):
            return ((x - min(x)) / (max(x) - min(x)))

        red = np.array([255, 0, 0])
        green = np.array([0, 255, 0])
        blue = np.array([0, 0, 255])
        red = np.round(red).astype("int64")
        blue = np.round(blue).astype("int64")
        green = np.round(green).astype("int64")

        col = green

        coloured_sents = []
        for ref in range(len(text_tok)):
            col_sent = []
            attn_weights = attention_weights[ref]
            attn_weights = norm(attn_weights) ** 5
            # attn_weights = softmax(np.array (attn_weights))

            # attn_weights = attn_weights [0,:]

            if (ref % 1000) == 0:
                print(ref)

            for i in range(0, len(text_tok[ref])):
                assert (len(attn_weights) == len(text_tok[ref]))

                alpha = np.array(attn_weights[i]) * 255

                alpha = np.round(alpha).astype("int64")

                col_word = cstr(text_tok[ref][i],
                                '#{:02x}{:02x}{:02x}{:02x}'.format(col[0], col[1], col[2], alpha))

                #         if (i % 10) == 0:
                #             col_sent.append(" <br> " + col_word)
                #         else:
                col_sent.append(col_word)

            coloured_sents.append(" ".join(col_sent))

        return (coloured_sents)

    def print_coloured_text (self, coloured_text):
        return (html_print(" <br> ".join(np.array(coloured_text))))#[ordering[0:10]]))

        # text_tok = self.texts_tok [text_tag]
        #
        # if subset is not None:
        #     text_tok = text_tok [subset]
        #     attention_weights = attention_weights [subset]
        #
        # def cstr(s, color='black'):
        #     return "<text style=background-color:{}>{}</text>".format(color, s)
        #
        # red = np.array([255, 0, 0])
        # blue = np.array([0, 255, 0])
        # red = np.round(red).astype("int64")
        # blue = np.round(blue).astype("int64")
        #
        # col = red
        #
        # coloured_sents = []
        # for ref in range (len (text_tok)):
        #     col_sent = []
        #     attn_weights = attention_weights [ref]
        #
        #     if (ref % 1000) == 0:
        #         print(ref)
        #
        #     for i in range(0, len(text_tok [ref])):
        #
        #         assert (len (attn_weights) == len (text_tok[ref]))
        #
        #         alpha = np.array(attn_weights[i]) * 255
        #
        #         alpha = np.round(alpha).astype("int64")
        #
        #         col_word = cstr(text_tok[ref][i],
        #                         '#{:02x}{:02x}{:02x}{:02x}'.format(col[0], col[1], col[2], alpha))
        #
        #         if (i % 10) == 0:
        #             col_sent.append(" <br> " + col_word)
        #         else:
        #             col_sent.append(col_word)
        #
        #     coloured_sents.append(" ".join(col_sent))
        #
        # return (coloured_sents)






        # self.tokenize (text_tag)
        # texts_tok = self.texts_tok[text_tag]
        #
        # dictionary = corpora.Dictionary(texts_tok)
        # self.gensim_dicts.update ({text_tag: dictionary})
        #
        # corpus = [dictionary.doc2bow(t) for t in texts_tok]
        #
        # tfidf = models.TfidfModel(corpus)
        #self.tfidf_models.update ({text_tag: tfidf})
        #
        # corpus_tfidf = tfidf[corpus]
        #
        # corpus_tfidf = gensim.matutils.corpus2csc (corpus_tfidf)

        #self.tf_idfs.update ({text_tag: corpus_tfidf})

    def transform_tf_idf(self, fit_model_tag, to_text_tag):
        text = self.texts [to_text_tag]
        tf_idf_model = self.tfidf_models [fit_model_tag]

        tf_idf_vec = tf_idf_model.transform (text)
        self.tf_idfs.update({to_text_tag: tf_idf_vec})

        # self.tokenize (to_text_tag)
        # texts_tok = self.texts_tok[to_text_tag]
        #
        # dictionary = self.gensim_dicts [fit_model_tag]
        # #dictionary = corpora.Dictionary(texts_tok)
        # #self.gensim_dicts.update ({to_text_tag: dictionary})
        #
        # corpus = [dictionary.doc2bow(t) for t in texts_tok]
        #
        # tfidf = self.tfidf_models [fit_model_tag] #models.TfidfModel(corpus)
        # #self.tfidf_models.update ({text_tag: tfidf})
        #
        # corpus_tfidf = tfidf[corpus]
        #
        # corpus_tfidf = gensim.matutils.corpus2csc (corpus_tfidf)
        #
        # self.tf_idfs.update ({to_text_tag: corpus_tfidf})


    def get_tf_idf (self, text_tag):
        return (self.tf_idfs [text_tag].toarray())


    def init_ltsm_classifier (self,
                              pretrain_traindf_tag,
                              pretrain_testdf_tag,
                              classif_traindf_tag,
                              classif_testdf_tag,
                              classif_trn_labs,
                              classif_tst_labs):

        lm_train_df = self.texts[pretrain_traindf_tag]
        lm_valid_df = self.texts[pretrain_testdf_tag]
        lm_train_df = pd.DataFrame (lm_train_df)
        lm_valid_df = pd.DataFrame(lm_valid_df)
        lm_train_df.columns = ["text"]
        lm_valid_df.columns = ["text"]

        class_train_df = self.texts[classif_traindf_tag]
        class_valid_df = self.texts[classif_testdf_tag]
        class_train_df = pd.DataFrame(class_train_df)
        class_valid_df = pd.DataFrame(class_valid_df)
        class_train_df.columns = ["text"]
        class_valid_df.columns = ["text"]
        class_train_df['label'] = classif_trn_labs
        class_valid_df['label'] = classif_tst_labs

        self.data_lm = TextLMDataBunch.from_df(path="./",
                                          train_df=lm_train_df,
                                          valid_df=lm_valid_df,
                                          text_cols="text",
                                          bs=64)

        self.data_clas = TextClasDataBunch.from_df(path="./", train_df=class_train_df, valid_df=class_valid_df, text_cols="text",
                                              label_cols="label", vocab=self.data_lm.train_ds.vocab, bs=32)

        self.ltsm_learn = language_model_learner(self.data_lm, pretrained_model=URLs.WT103_1, drop_mult=0.5)
        self.ltsm_learn.save_encoder('ft_enc')

    def pretrain_ltsm (self):
        self.ltsm_learn.fit_one_cycle(1, 1e-2)
        self.ltsm_learn.unfreeze()
        self.ltsm_learn.fit_one_cycle(1, 1e-3)

    def train_ltsm_classifier (self, nepoch = 5, lrate = 1e-2, continue_training = False):
        if not continue_training:
            self.ltsm_learn = text_classifier_learner(self.data_clas, drop_mult=0.5)
            self.ltsm_learn.load_encoder('ft_enc')

        self.ltsm_learn.fit_one_cycle(1, 1e-2)

        self.ltsm_learn.freeze_to(-2)
        self.ltsm_learn.fit (nepoch, lrate)

def load (filename):
    with open(filename, 'rb') as input:
        loaded_class = pickle.load(input)
    return (loaded_class)