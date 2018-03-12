import random
import time
import math
import torchtext.vocab as vocab
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from Utils import masked_cross_entropy
from modele import EncoderRNN, DecoderStep
from prepare_data import indextoword, indexes_from_sentence, wordtoindex, pairs_test, random_batch, pairs_train

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

"""**TRAIN & EVALUATE**"""


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH, E_hist=None):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Share embedding
    decoder.embedding.weight = encoder.embedding.weight

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))
    hidden_history_decoder = decoder_hidden[-1].unsqueeze(0)

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn, E_hist, hidden_history_decoder = decoder(
            decoder_input, decoder_hidden, encoder_outputs, E_hist, t, hidden_history_decoder, input_batches
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc


def evaluate(input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq.split())]
    input_seqs = [indexes_from_sentence(wordtoindex, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    hidden_history_decoder = decoder_hidden[-1].unsqueeze(0)
    E_hist = None
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention, E_hist, hidden_history_decoder = decoder(
            decoder_input, decoder_hidden, encoder_outputs, E_hist, di, hidden_history_decoder, input_batches
        )
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(indextoword[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate_randomly():
    [input_sentence, target_sentence] = random.choice(pairs_test)
    evaluate_and_show_attention(input_sentence, target_sentence)


def evaluate_and_show_attention(input_sentence, target_sentence=None):
    output_words, attentions = evaluate(input_sentence)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)

    show_attention(input_sentence, output_words, attentions)


"""**Vizualisation**"""


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def show_losses(y1, y2, y3):
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.plot(y1, label='data 1')
    ax2.plot(y2, label='data 2')
    ax3.plot(y3, label='data 3')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('encoder grad')
    ax1.set_title('GRAD ENCODER WITH EPOCH')
    ax1.legend()
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('epochs')
    ax2.set_title('GRAD DECODER WITH EPOCH')
    ax2.legend()
    ax3.set_xlabel('epochs')
    ax3.set_title('LOSS WITH EPOCH')
    ax3.legend()
    plt.show()


"""**MAIN**"""

# Configure models
attn_model = 'concat'
hidden_size = 500
n_layers = 2
dropout = 0.1
batch_size = 50
embed_size = 100
# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 50000
epoch = 0
plot_every = 20
print_every = 100
evaluate_every = 1000
# config type model
encoder_temporal = True
decoder_attention_bol = True
pointeur_bolean = True

# Initialize models
encoder = EncoderRNN(len(indextoword), embed_size, hidden_size, n_layers, dropout=dropout)
decoder = DecoderStep(hidden_size, embed_size, len(indextoword), n_layers, dropout_p=dropout, temporal=encoder_temporal,
                      de_att_bol=decoder_attention_bol, point_bol=pointeur_bolean)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

# Begin!
ecs = []
dcs = []
eca = 0
dca = 0

while epoch < n_epochs:
    epoch += 1

    print(epoch)

    # print (epoch)
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, pairs_train, wordtoindex)
    # Run the train function
    loss, ec, dc = train(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion
    )

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    eca += ec
    dca += dc

    if epoch == 30:
        evaluate_randomly()

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (
        time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)
        show_losses(ecs, dcs, plot_losses)

    if epoch % evaluate_every == 0:
        evaluate_randomly()

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

        # TODO: Running average helper
        ecs.append(eca / plot_every)
        dcs.append(dca / plot_every)

        eca = 0
        dca = 0
        plot_loss_total = 0
