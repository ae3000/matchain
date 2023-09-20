import logging
import multiprocessing
from typing import Union

import gensim.models
import gensim.models.callbacks
from tqdm import tqdm


def train(sentences: Union[str, list], word2vec_params: dict,
          file_embedding: str):

    epochs = word2vec_params['epochs']

    gensim_params = {}
    allowed_keys = [
        'epochs', 'min_count', 'negative', 'sample', 'sg', 'vector_size',
        'window', 'workers'
    ]
    for key, value in word2vec_params.items():
        if key in allowed_keys:
            gensim_params[key] = value

    if not gensim_params.get('workers'):
        cpu_count = multiprocessing.cpu_count()
        gensim_params['workers'] = cpu_count

    logging.info("training gensim's word2vec with params=%s", gensim_params)

    if isinstance(sentences, str):
        gensim_params['corpus_file'] = sentences
    else:
        gensim_params['sentences'] = sentences

    # if ascii=True, progress bar can be serialized for logging without any unicode error
    with tqdm(total=epochs, ascii=False) as pbar:
        callback = Callback(pbar)
        model = gensim.models.Word2Vec(callbacks=[callback],
                                       compute_loss=True,
                                       **gensim_params)

    #message = str(pbar).replace('#', '').replace('||', '|#####|')
    #logging.info('finished training, %s', message)
    #logging.debug('losses=%s', callback.loss_dict)
    logging.info('finished training, losses=%s', callback.loss_dict)

    if file_embedding:
        model.wv.save_word2vec_format(file_embedding, binary=False)


class Callback(gensim.models.callbacks.CallbackAny2Vec):

    def __init__(self, pbar):
        self.epoch = 0
        self.last_loss = 0
        self.pbar = pbar
        self.loss_dict = {}

    def on_epoch_end(self, model):
        # https://stackoverflow.com/questions/52038651/loss-does-not-decrease-during-training-word2vec-gensim
        loss = model.get_latest_training_loss()
        diff = loss - self.last_loss
        self.last_loss = loss
        self.pbar.set_description(f'loss:{diff:.2f}')
        self.pbar.update(1)
        self.epoch += 1
        self.loss_dict[self.epoch] = diff


def run(sentences: Union[str, list], word2vec_params: dict,
        file_embedding: str):
    train(sentences, word2vec_params, file_embedding)
