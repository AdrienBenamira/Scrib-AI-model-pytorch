import math
import torchtext.vocab as vocab
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from prepare_data import pretrained_weight

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
MIN_LENGTH_ARTICLE = 10
MIN_LENGTH_SUMMARY = 5
MAX_LENGTH = 50
dim = 100
glove = vocab.GloVe(name='6B', dim=dim)
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()

        # Define parameters
        self.input_size = input_size  # V Taille Vocabulary (can be different, here not)
        self.hidden_size = hidden_size  # H
        self.embed_size = embed_size  # E
        self.n_layers = n_layers  # L (1 per default)
        self.dropout = dropout  # 0.5 per default

        # Define layers
        self.embedding = nn.Embedding(input_size, embed_size)  # Init (V,E)
        self.embedding.weight = nn.Parameter(pretrained_weight)  # Init with glove

        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout,
                          bidirectional=True)  # Init (E,H,L, Bidirectionnel!)

    def forward(self, input_seqs, input_lengths, hidden=None):
        '''
        :param input_seqs:
            Variable of shape (T,B), T is the number of words in the longuest sentence, B is Batchsize. Contening the indexing of the words reference to the voc
        :param input_lengths:
            list of integers (len=B) which reprensents the number of words in sequence for each batch. Normally Max(input_lengths)=T
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,H)
            last hidden stat of RNN(L*bidirectionnal,B,H)
        '''
        embedded = self.embedding(input_seqs)  # (T,B,E)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,
                                                         input_lengths)  # cf doc pytorch : take embedding and input_length. Ready to go
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden  # (T,B,H),(L*bidirectionnal,B,H) | bidirectionnal=2 here


class Attn(nn.Module):
    def __init__(self, method, hidden_size, temporal=False):
        super(Attn, self).__init__()

        # Define parameters
        self.method = method  # 2 methods cf publi
        self.hidden_size = hidden_size  # H
        self.temporal = temporal  # Temporal attention encoder
        self.softmax = nn.Softmax()

        # Define layers
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)  # Init(2*H,H)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, E_history=None):
        '''
        :param hidden:
            (B,H)
        :param encoder_outputs:
            (T,B,H) can be also hidden decoder accumulation over time (t+1,B,H), depends on attention encoder or decoder
        :param E_history:
           Encoder history use only if intra temporal attention. Init with None, then (t,B,T)
        :returns:
            attn_energies which is alpha
        '''

        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(H, encoder_outputs)

        if self.temporal:
            if E_history is None:
                E_history = attn_energies.unsqueeze(0)
            else:
                E_history = torch.cat([E_history, attn_energies.unsqueeze(0)], 0)
                hist = E_history.view(-1, this_batch_size * max_len).t()
                attn_energies = self.softmax(hist)[:, -1].contiguous().view(this_batch_size, max_len)
            return F.softmax(attn_energies).unsqueeze(1), E_history
        else:
            # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
            return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            input = torch.cat([hidden, encoder_output], 2)
            energy = F.tanh(self.attn(input))  # [B*T*2H]->[B*T*H]
            energy = energy.transpose(2, 1)  # [B*H*T]
            v = self.v.repeat(encoder_output.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
            energy = torch.bmm(v, energy)  # [B*1*T]
            return energy.squeeze(1)  # [B*T]


class DecoderStep(nn.Module):
    """
    Class to call for each generation of word
    """

    def __init__(self, hidden_size, embed_size, output_size, n_layers, temporal=True, de_att_bol=True, point_bol=True,
                 dropout_p=0.1):
        super(DecoderStep, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size  # H
        self.output_size = output_size  # V
        self.n_layers = n_layers  # L
        self.dropout_p = dropout_p  # 0.1 per default
        self.temporal = temporal  # bolean to use intra temporal attention on input sequence cf. Temporal attention model for neural machine translation. arXiv preprint arXiv:1608.02927, 2016
        self.decoder_attention_bolean = de_att_bol  # bolean to use intra decoder attention
        self.pointer_boloan = point_bol

        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)  # Init(V,E)
        self.dropout = nn.Dropout(dropout_p)
        self.attn_encoder = Attn('concat', hidden_size, temporal)  # Init(methode score, H, bolean temporal), cf class
        if self.decoder_attention_bolean:
            self.attn_decoder = Attn('concat', hidden_size, temporal=False)
            self.gru = nn.GRU(2 * hidden_size + embed_size, hidden_size, n_layers,
                              dropout=dropout_p)  # init(3*H+E,H,L)
            self.out = nn.Linear(hidden_size * 3, output_size)  # Wout(3H,V) case [1] and [2]
            self.out_proba = nn.Linear(hidden_size * 3, 1)
        else:
            self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)  # init(2*H+E,H,L)
            self.out = nn.Linear(hidden_size * 2, output_size)  # Wout(2H,V) case [1] and [2]
            self.out_proba = nn.Linear(hidden_size * 2, 1)

    def forward(self, word_input, last_hidden, encoder_outputs, E_hist, t, hd_history, input_batches):
        '''
        :param word_input:
            tensor with SOS_Token length B
        :param last_hidden:
            Last hidden of the decoder, initialization with last hidden encoder (L,B,H)
        :param encoder_outputs:
            encoder output (T,B,H)
        :param E_hist:
            Encoder history use only if intra temporal attention. Init with None, then (1,B,T)
        :param t:
            number of time generate words [0 to max_length sequence]
        :param hd_history:
            hidden decoder accumulation over time (t+1,B,H)
        :param input_batches:
            input de l'encoder to be able to point to the entry (V,B)
        :returns:
            output decoder : max proba is the word to generate (B,V)
            hidden : will be last hidden next time
            alpha : to plot during the evaluation
            E_history : will be E_hist next time
            hd_history : will be hd_history next time
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1)  # (1,B,E)
        word_embedded = self.dropout(word_embedded)

        # Calculate attention weights -temporal or not- of encoder (alpha) and apply to encoder outputs (context_encoder)
        if self.temporal:
            alpha, E_history = self.attn_encoder(last_hidden[-1], encoder_outputs, E_hist)  # (B,1,T) (1,B,T)
        else:
            E_history = None  # None
            alpha = self.attn_encoder(last_hidden[-1], encoder_outputs)  # (B,1,T) alpha will be use later
        context_encoder = alpha.bmm(encoder_outputs.transpose(0, 1))  # (B,1,H)
        context_encoder = context_encoder.transpose(0, 1)  # (1,B,H) context with the encoder.

        # attention on decoder and RNN input
        if self.decoder_attention_bolean:
            if t:  # Recurrence
                alpha_d = self.attn_decoder(last_hidden[-1], hd_history)  # (B,1,t)
                context_decoder = alpha_d.bmm(hd_history.transpose(0, 1))  # (1,B,H)
                context_decoder = context_decoder.transpose(0, 1)  # (1,H,B)
                hd_history = torch.cat([hd_history, last_hidden[-1].unsqueeze(0)], dim=0)  # (t+1,B,H)
            else:  # Initialisation
                context_decoder = Variable(torch.zeros(1, word_embedded.size()[1], self.hidden_size))  # init to zero
                if USE_CUDA:
                    context_decoder = context_decoder.cuda()
            rnn_input_old = torch.cat((word_embedded, context_encoder), 2)
            rnn_input = torch.cat((rnn_input_old, context_decoder), 2)  # (1,B,H+E+t) idem
        else:
            # Combine embedded input word and attended context, run through RNN
            rnn_input = torch.cat((word_embedded, context_encoder), 2)

        # RNN
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,H)->(B,H)
        context_encoder = context_encoder.squeeze(0)
        output_concat = torch.cat((output, context_encoder), 1)

        # pointer
        if self.pointer_boloan:
            if self.decoder_attention_bolean:
                context_decoder = context_decoder.squeeze(0)
                output_concat_v2 = torch.cat((output_concat, context_decoder), 1)
                p_yt_sachant_paspointer = F.softmax(self.out(output_concat_v2))
                p_pointer = F.sigmoid(self.out_proba(output_concat_v2))
            else:
                p_yt_sachant_paspointer = F.softmax(self.out(output_concat))
                p_pointer = F.sigmoid(self.out_proba(output_concat))
            # Calcul proba finale
            alpha_interet = alpha.squeeze(1).transpose(0, 1)  # (T,B)
            batch_size = alpha_interet.size()[1]
            mask = self.alpha_to_mask(batch_size, alpha_interet, input_batches)  # (V,B), le pointeur cf equation 10

            proba_final = torch.mul(p_yt_sachant_paspointer, (1 - p_pointer)) + torch.mul(mask,
                                                                                          p_pointer)  # (B,V) equation 12
            output = torch.log(proba_final)
        else:
            if self.decoder_attention_bolean:
                context_decoder = context_decoder.squeeze(0)
                output_concat_v2 = torch.cat((output_concat, context_decoder), 1)
                output = F.log_softmax(self.out(output_concat_v2))
            else:
                output = F.log_softmax(self.out(output_concat))

        return output, hidden, alpha, E_history, hd_history

    def alpha_to_mask(self, batch_size, alpha_interet, input_batches):
        # Here the goal is to transform alpha_interet(T,B) into mask(V,B)
        mask = Variable(torch.zeros(batch_size, self.output_size))
        if USE_CUDA:
            mask = mask.cuda()
        for i in range(batch_size):
            for j in range(alpha_interet.size()[0]):
                mask[i, input_batches[j, i].data[0]] = alpha_interet[j, i][0]
        return mask
