# modification of model from https://github.com/avisingh599/visual-qa
from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Merge, Dense
from keras.layers import concatenate,Concatenate,Dot
from keras.layers import Add
from keras.models import Model
def VQA_MODEL():
    image_feature_size          = 4096
    word_feature_size           = 300
    number_of_LSTM              = 3
    number_of_hidden_units_LSTM = 512
    max_length_questions        = 30
    number_of_dense_layers      = 3
    number_of_hidden_units      = 1024
    activation_function         = 'tanh'
    dropout_pct                 = 0.5


    # Image model
    model_image = Sequential()
    model_image.add(Reshape((image_feature_size,), input_shape=(image_feature_size,)))

    # Language Model
    model_language = Sequential()
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(max_length_questions, word_feature_size)))
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True))
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=False))

    # combined model
    x=concatenate([model_language.output,model_image.output],axis=1)

    for _ in xrange(number_of_dense_layers):
        x=Dense(number_of_hidden_units, kernel_initializer='uniform')(x)
        x=Activation(activation_function)(x)
        x=Dropout(dropout_pct)(x)

    model_output=Dense(1000)(x)
    model_output=Activation('softmax')(model_output)


    final=Model([model_language.input,model_image.input],model_output)

    print final.summary()
    return final

model=VQA_MODEL()
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)





