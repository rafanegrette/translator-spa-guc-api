from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import re
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

app = Flask(__name__)
cors = CORS(app, resources={r'/*': {'origins': '*'}})
app.config['CORS_HEADERS'] = 'Content-Type'

path_to_file = 'files/way_spa_master.xlsx';

corpus = pd.read_excel(path_to_file, engine='openpyxl')

path_to_file = 'files/frases.xlsx';

corpus_phrase = pd.read_excel(path_to_file)

corpus = pd.concat([corpus, corpus_phrase], ignore_index = True)

path_to_file = 'files/phrases_little_prince.xlsx';

litte_prince_corpus = pd.read_excel(path_to_file, engine='openpyxl')

corpus = pd.concat([corpus, litte_prince_corpus], ignore_index = True)

path_to_file = 'files/opus_result.xlsx';
opus_corpus = pd.read_excel(path_to_file, engine='openpyxl')


corpus = pd.concat([corpus, opus_corpus], ignore_index = True)

path_to_file = 'files/WayuuDict.xlsx';
dict_corpus = pd.read_excel(path_to_file, engine='openpyxl')


corpus = pd.concat([corpus, dict_corpus], ignore_index = True)

index_sorted = corpus.spa.str.len().sort_values(ascending=True).index
corpus = corpus.reindex(index_sorted)
corpus = corpus.reset_index(drop=True)

def unicode_to_ascii(s):
    r = ""
    for c in s:
        if c == 'á':
            r += 'a'
        elif c == 'é':
            r += 'e'
        elif c == 'í':
            r += 'i'
        elif c == 'ó':
            r += 'o'
        elif c == 'ú':
            r += 'u'
        else:
            r += c
    return r
  
def preprocess_sentence(w):
  try:
      if pd.isnull(w) or pd.isna(w):
        #print('1 IF: {}'.format(w))
        return None
      w = unicode_to_ascii(str(w).lower().strip())
      w = re.sub(r"([?.,¿])", r" \1 ", w)
      w = re.sub(r'[" "]+üÜ', " ", w)
      w = re.sub(r"[^a-zA-ZʼüÜ']+", " ", w)
      w = w.strip()
      if w.isspace() or "\n" in w or not w:
        #print('2 IF: {}'.format(w))
        return None
      w = '<start> ' + w + ' <end>'
  except Exception as e:
    print(e)    
  return w

def to_list(doc):
    return doc.values.tolist()

def filterOtliers(wy, sp):
    filteredSpa = []
    filteredWay = []
    for forWay, forSpa in zip(wy, sp):        
        if len(forSpa.split()) < 15 and len(forWay.split()) < 15 and abs(len(forSpa.split()) - len(forWay.split())) < 6:
            filteredSpa.append(forSpa)
            filteredWay.append(forWay)    
    return filteredWay, filteredSpa


def create_dataset(corpus):
    sp_phrases = []
    wy_phrases = []
    phrases = {}
    corpusList = to_list(corpus)
    count = 0
    for phrase in corpusList:
        try:
            sentence_0 = preprocess_sentence(phrase[0])
        
            try:
                sentence_1 = preprocess_sentence(phrase[1])
            except IndexError:
                sentence_2 = None
            try:
                sentence_2 = preprocess_sentence(phrase[2])
            except IndexError:
                sentence_2 = None
                
            if sentence_0 is None:
                count= 1 + count
                continue
            if sentence_1 is None:
                if sentence_2 is None:
                    continue
                else: 
                    phrases[sentence_0] = sentence_2
            else:
                phrases[sentence_0] = sentence_1
        except Exception as e:
            print("Error 0: {}".format(phrase))
            print(e)
        
    wy, sp = filterOtliers(list(phrases.values()), list( dict.fromkeys(phrases)))
    return wy, sp
    
wy, sp = create_dataset(corpus)
sp_choices = [i[8:][:-6] for i in sp]
wy_choices = [i[8:][:-6] for i in wy]

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

  return tensor, lang_tokenizer

def load_dataset(dataframe):
  targ_lang, inp_lang = create_dataset(dataframe)
  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(corpus)

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d -----> %s" % (t, lang.index_word[t]))

convert(inp_lang, input_tensor_train[0])
convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len (targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    
  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
  
  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

attention_layer = BahdanauAttention(15)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)
  
  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)
    return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)


decoder.load_weights('model2/decoder.h5')
encoder.load_weights('model2/encoder.h5')

def getValueFromDictionary(sentence):
    dict_result = "";
    try:        
        index_value_dict = sp_choices.index(process.extractOne(sentence, sp_choices)[0])
        dict_result = sp_choices[index_value_dict]  + " : " + wy_choices[index_value_dict]
    except:
        dict_result = "not in dictionary"
    return dict_result


def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentence(sentence)

  try:      
      inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  except:
    return "No result", sentence, getValueFromDictionary(sentence)
  
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1,units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
    attention_weights = tf.reshape(attention_weights, (-1,))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()
    
    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, getValueFromDictionary(sentence)

    result += targ_lang.index_word[predicted_id] + ' '

    dec_input = tf.expand_dims([predicted_id], 0)
    
  return result, sentence, getValueFromDictionary(sentence)

def predict(event, context):
    body = {
        "message": "OK",
    }

    phrase = event['text']
    predicted, _, dict_phrase =evaluate(phrase)

    try:
        result = phrase
        response = jsonify({
            "statusCode": 200,
            "body": {
                "predict": predicted,
                "dictionary": dict_phrase
            }
        })
    except Exception as ex:
        response = jsonify({
            "statusCode": 500,
            "body": "Error con :" + str(ex)
        })

    return response

@app.route('/', methods=['POST'])
@cross_origin()
def do_main():
    json_data = ""
    if request.method == 'POST':
        json_data = request.get_json()

    response = predict(json_data, None)
    #body = json.loads(response['body'])
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)