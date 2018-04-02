def get_vocab(scribAPI, vocab_size, pad_token=0, unk_token=3, sos_token=1, eos_token=2):
    """
    get word to index and index to word
    :param scribAPI: ScribAI API object
    :param vocab_size: size of the wanted vocab
    """
    vocab = scribAPI.get_vocab(limit=vocab_size)
    vocab_words = list(map(lambda x: x['word'], vocab))
    vocab_counts = list(map(lambda x: x['count'], vocab))
    word2index = {"PAD":pad_token, "SOS":sos_token,"EOS":eos_token,"UNK": unk_token}
    word2count = {"UNK": 1}
    index2word = {pad_token: "PAD", sos_token: "SOS", eos_token: "EOS", unk_token: "UNK"}
    next_token = max([pad_token, unk_token, sos_token, eos_token]) + 1
    for i, word in enumerate(vocab_words):
        try:
            g = glove.vectors[glove.stoi[word]]
            word2index[word] = next_token
            index2word[next_token] = word
            word2count[word] = vocab_counts[i]
            next_token += 1
        except:
            word2count["UNK"] += 1
    return word2index, word2count, index2word    
    

def get_vect_from_word(word):
    return glove.vectors[glove.stoi[word]]

def tokenize_article_in_words(text_article):
    sentences = [word_tokenize(t) for t in sent_tokenize(text_article)]
    words = []
    for sentence in sentences:
        words.extend(sentence)
    return words

def words_into_vect(words):
    vector = None
    for i, word in enumerate(words):
        if i == 0:
          try:
            vector = get_vect_from_word(word.lower())
          except Exception:
            vector = get_vect_from_word('unk')
        elif i == 1:
          try:
            vector = torch.stack((vector, get_vect_from_word(word.lower())), 1)
          except Exception:
            vector = torch.stack((vector, get_vect_from_word('unk')), 1)
        else:
          try:
            vector = torch.cat((vector, get_vect_from_word(word.lower())), 1)
          except Exception:
            vector = torch.cat((vector, get_vect_from_word('unk')), 1)
    return vector

def create_vocab_from_articles(A):
    word2index = {"PAD":0, "SOS":1,"EOS":2,"UNK": 3}
    word2count = {"UNK": 1}
    index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
    n_words = 4  # Count default tokens
    compteur_general=0
    for i, article in enumerate(A):
        words=tokenize_article_in_words(article)
        for word in words:
            compteur_general+=len(words)
            if word not in word2index:
                try:
                    get_vect_from_word(word.lower())
                    word2index[word] = n_words
                    word2count[word] = 1
                    index2word[n_words] = word
                    n_words += 1
                except Exception:
                    word2count["UNK"] += 1
            else:
                word2count[word] += 1
    return word2index, word2count, index2word, compteur_general

def create_ini_embedding(wordtoindex):
    sample=wordtoindex.keys()
    return words_into_vect(sample)

  
def pairs_and_filterpairs(articles,titles,word2index,m,n, seuil=3):
    pairs=[]
    compteur_train=0
    for k, article in enumerate(articles):
        compteur=0
        words=tokenize_article_in_words(article)
        words_target=tokenize_article_in_words(titles[k])
        if len(words) >= m and len(words_target) >= n:
            for word in words:
                try:
                    word2index[word]
                except Exception:
                    compteur=compteur+1
                if word=='UNK':
                    compteur=compteur+1
            for word in words_target:
                try:
                    word2index[word]
                except Exception:
                    compteur=compteur+1
                if word=='UNK':
                    compteur=compteur+1
            if (100*float(compteur)/float(len(words)+len(words_target)))<seuil:
              pairs.append([article,titles[k]])
    return pairs


def indexes_from_sentence(word2index, sentence):
    ind=[]
    for word in sentence.split(' '):
        try:
            ind.append(word2index[word])
        except Exception:
            ind.append(3)
    return ind + [EOS_token]
  
def indexes_from_sentence_target(word2index, sentence,input_seq):
    target_normal=[]
    target_context=[]
    target_pointer=[]
    words_input=input_seq.split(' ')
    for word in sentence.split(' '):
        try:
            target_normal.append(word2index[word])
            target_pointer.append(0)
            target_context.append(0)
        except Exception:
            target_normal.append(UNK_token)
            try:
              index_element = words_input.index(word)
              target_context.append(index_element)
              p=1
            except Exception:
              target_context.append(0)
              p=0
            target_pointer.append(p)
    return target_normal + [EOS_token], target_context +[EOS_token], target_pointer
  
# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq
def pad_seq_l(seq, max_length):
    seq += ['PAD' for i in range(max_length - len(seq))]
    return seq

def random_batch(batch_size, pairs,word2index):
    input_seqs = []
    input_seqs_letters=[]
    target_seqs_voc = []
    target_seqs_context = []
    target_seqs_pointer = []
    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs_letters.append(pair[0].split(' '))
        input_seqs.append(indexes_from_sentence(word2index, pair[0]))
        tn,tc,tp=indexes_from_sentence_target(word2index, pair[1],pair[0])
        target_seqs_voc.append(tn)
        target_seqs_context.append(tc)
        target_seqs_pointer.append(tp)
    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs_letters, input_seqs, target_seqs_voc,target_seqs_context, target_seqs_pointer), key=lambda p: len(p[0]), reverse=True)
    input_seqs_letters, input_seqs, target_seqs_voc,target_seqs_context, target_seqs_pointer = zip(*seq_pairs)
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs_voc]
    target_padded_voc = [pad_seq(s, max(target_lengths)) for s in target_seqs_voc]
    target_padded_context = [pad_seq(s, max(input_lengths)) for s in target_seqs_context]
    target_padded_pointer = [pad_seq(s, max(target_lengths)) for s in target_seqs_pointer]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var_voc = Variable(torch.LongTensor(target_padded_voc)).transpose(0, 1)
    target_var_context= Variable(torch.LongTensor(target_padded_context)).transpose(0, 1)
    target_var_pointer = Variable(torch.LongTensor(target_padded_pointer)).transpose(0, 1)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var_voc = target_var_voc.cuda()
        target_var_context=target_var_context.cuda()
        target_var_pointer=target_var_pointer.cuda()
    
    return input_var, input_lengths, target_var_voc, target_lengths,target_var_context, target_var_pointer,input_seqs_letters

def random_batch_three(batch_size, articles, article_refs, articles2, article_refs2, scores, word2index):
    input_seqs = []
    input_seqs_2 = []
    input_seqs_refs = []
    input_seqs_refs_2 = []
    target_seqs = []
    # Choose random pairs
    for i in range(batch_size):
        k = random.randint(0, len(articles)-1)
        article, article_ref = indexes_from_sentence(word2index, articles[k]), indexes_from_sentence(word2index, article_refs[k])
        article2, article_ref2 = indexes_from_sentence(word2index, articles2[k]), indexes_from_sentence(word2index, article_refs2[k])
        input_seqs.append(article)
        input_seqs_2.append(article2)
        input_seqs_refs.append(article_ref)
        input_seqs_refs_2.append(article_ref2)
        target_seqs.append(scores[k])
    # Zip into pairs, sort by length (descending), unzip
    order_batches1 = [k for k in range(len(input_seqs))]
    order_batches2 = [k for k in range(len(input_seqs_2))]
    seqs1 = sorted(zip(input_seqs, input_seqs_refs, order_batches1), key=lambda p: len(p[0]) + len(p[1]), reverse=True)
    seqs2 = sorted(zip(input_seqs_2, input_seqs_refs_2, order_batches2), key=lambda p: len(p[0]) + len(p[1]), reverse=True)
    input_seqs, input_seqs_refs, order_batches1 = zip(*seqs1)
    input_seqs_2, input_seqs_refs_2, order_batches2 = zip(*seqs2)
    input_lengths = [len(s) for s in input_seqs]
    input_lengths_2 = [len(s) for s in input_seqs_2]
    input_seqs = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    input_seqs_2 = [pad_seq(s, max(input_lengths_2)) for s in input_seqs_2]
    input_seqs_refs_length = [len(s) for s in input_seqs_refs]
    input_seqs_refs_length_2 = [len(s) for s in input_seqs_refs_2]
    input_seqs_refs = [pad_seq(s, max(input_seqs_refs_length)) for s in input_seqs_refs]
    input_seqs_refs_2 = [pad_seq(s, max(input_seqs_refs_length_2)) for s in input_seqs_refs_2]
    target_seqs = list(target_seqs)

    return input_seqs, input_lengths, input_seqs_refs, input_seqs_refs_length, order_batches1, input_seqs_2, input_lengths_2, input_seqs_refs_2, input_seqs_refs_length_2, order_batches2, target_seqs


