import torch
from torch.autograd import Variable
from nltk.tokenize import sent_tokenize, word_tokenize


def sequence_to_words(glove, sequence):
    words = []
    for k in range(sequence.size(0)):
        words.append(glove.reverse(sequence[k]))
    return ' '.join(words)


def get_word_index(words, start_of_line='<s>', end_of_line='<e>', unknown='<unk>', pad_word='<pad>'):
    index_to_word = {}
    word_to_index = {}
    w = [{'word': start_of_line}, {'word': end_of_line}, {'word': unknown}, {'word': pad_word}]
    w.extend(words)
    for i, word in enumerate(w):
        index_to_word[i] = word["word"]
        word_to_index[word["word"]] = i
    return index_to_word, word_to_index


def get_word_from_index(index, index_to_word, unknown='<unk>'):
    if index in index_to_word.keys():
        return index_to_word[index]
    return unknown


def get_index_from_word(word, word_to_index, unknown='<unk>'):
    if word in word_to_index.keys():
        return word_to_index[word]
    return word_to_index[unknown]


def strip_article(article):
    article = article.replace('\\n', '')
    return article


def get_vector_article(article, word_to_index, start='<s>', end='<e>', unknown='<unk>'):
    article = strip_article(article)
    words = [start] + tokenize(article)
    words.append(end)
    return torch.LongTensor([get_index_from_word(word, word_to_index, unknown) for word in words])


def pad_sequence(articles, length, word_to_index, pad_token='<pad>'):
    result = torch.LongTensor(len(articles), max(length)).fill_(word_to_index[pad_token])
    for i, article in enumerate(articles):
        result[i, :len(article)] = article
    return result


def tokenize(text_article):
    """
    :param text_article:
    :return:
    """
    sentences = [word_tokenize(t) for t in sent_tokenize(text_article)]
    words = []
    for sentence in sentences:
        words.extend(sentence)
    return words


def gliding_window(input, window_size, batch_size=1):
    """
    Make a window of input
    :param input:
    :param window_size:
    :param batch_size:
    :return:
    """
    seq_len = input.size(1)
    input_size = input.size(2)
    out_seq_len = seq_len - window_size + 1
    windows = torch.zeros(batch_size, out_seq_len, window_size, input_size)
    for k in range(out_seq_len):
        for batch in range(batch_size):
            for num_window in range(window_size):
                windows[batch, k, num_window, :] = input[batch, k + num_window]
    return windows


def eval(model, article, word_to_index=None, start='<e>', end='<e>', unknown='<unk>'):
    vector = get_vector_article(article, word_to_index, start, end, unknown) if word_to_index is not None else article
    vector = Variable(torch.unsqueeze(vector, 0))
    return model(vector)


def train(rouge, encoder, decoder, batch, SOS_token, EOS_token, hidden_size, encoder_optimizer, decoder_optimizer,
          window_size=10, batch_size=1, max_size=10):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input = gliding_window(batch['article'], window_size, batch_size)
    hidden = encoder.init_hidden()
    n_sequences = input.size(1)
    outputs = Variable(torch.zeros(n_sequences, 1, 1, hidden_size))
    if torch.cuda.is_available():
        outputs.cuda()
    # Encoder
    for n_seq in range(n_sequences):
        input_seq = Variable(input[:, n_seq])
        if torch.cuda.is_available():
            input_seq.cuda()
        output, hidden = encoder(input_seq, hidden)
        outputs[n_seq] = output
    # Decoder
    SOS = SOS_token.unsqueeze(0)
    EOS = EOS_token.unsqueeze(0)
    decoder_input = Variable(SOS)
    if torch.cuda.is_available():
        decoder_input.cuda()

    output = decoder_input
    k = 0
    outputs_decoder = torch.zeros(max_size, SOS_token.size(0))
    hiddens_decoder = []
    while torch.max(output.data.cpu() != EOS) and k < max_size:
        if torch.cuda.is_available():
            output, hidden, attention = decoder(output.cuda(), hidden, outputs.cuda())
        else:
            output, hidden, attention = decoder(output, hidden, outputs)
        outputs_decoder[k] = output.view(-1).data
        # outputs_decoder.append(output.view(-1).data.numpy())
        hiddens_decoder.append(hidden.data.cpu().numpy())
        k += 1

    metric = rouge(outputs_decoder, batch['article'][0])
    loss = Variable(torch.exp(-metric), requires_grad=True)
    if torch.cuda.is_available():
        loss.cuda()
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data
