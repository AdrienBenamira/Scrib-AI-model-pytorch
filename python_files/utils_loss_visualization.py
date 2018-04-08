import time
import torch
import torch.nn.functional as F
import math
import matplotlib as plt
from matplotlib import ticker
from torch.autograd import Variable
USE_CUDA =torch.cuda.is_available()
print('On utilise le GPU, '+str(USE_CUDA)+' story')



def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if USE_CUDA:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    if USE_CUDA:
      seq_length_expand=seq_length_expand.cuda()
    return seq_range_expand < seq_length_expand

def masked_cross_entropy(logits, target, length):
    length = Variable(torch.LongTensor(length))
    if USE_CUDA:
      length=length.cuda()
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)

    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def evaluate_randomly():
    [input_sentence, target_sentence] = torch.random.choice(pairs_train)
    evaluate_and_show_attention(input_sentence, target_sentence)
    
def evaluate_and_show_attention(input_sentence, target_sentence=None):
    output_words, attentions = Seq2SEq_main_model.evaluate(input_sentence)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)
    
    show_attention(input_sentence, output_words, attentions)

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
    
def show_losses(y1,y2,y3):
    fig = plt.figure(figsize=(18, 6))
    if y1 is not None:
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(y1, label='data 1')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('encoder grad')
        ax1.set_title('GRAD ENCODER WITH EPOCH')
        ax1.legend()
    if y2 is not None:
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(y2, label='data 2')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('epochs')
        ax2.set_title('GRAD DECODER WITH EPOCH')
        ax2.legend()
    if y3 is not None:
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(y3, label='data 3')
        ax3.set_xlabel('epochs')
        ax3.set_title('LOSS WITH EPOCH')
        ax3.legend()
    plt.show()

