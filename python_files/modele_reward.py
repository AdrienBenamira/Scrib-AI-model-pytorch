import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from modele_seq2seq import EncoderRNN
USE_CUDA =torch.cuda.is_available()
print('On utilise le GPU, '+str(USE_CUDA)+' story')



class RewardPredictor(nn.Module):
  
  def __init__(self, input_size, embed_size, hidden_size, indextoword,
                 wordtoindex, num_layer_encoder=1,
                 dropout=0.5, SOS_token=1, EOS_token=2, PAD_token=0, batch_size=5, article_max_size=500):
    super(RewardPredictor, self).__init__()
    """
    :param input_size:
    :param embed_size:
    :param hidden_size:
    :param indextoword:
    :param wordtoindex
    :param num_layer_encoder:
    :param dropout:
    :param SOS_token:
    :param EOS_token:
    :param PAD_token:
    :param batch_size:
    :param article_max_size:
    """
    super(RewardPredictor, self).__init__()
    self.encoder = EncoderRNN(input_size, embed_size, hidden_size,pretrained_weight, n_layers=num_layer_encoder, dropout=dropout)
    
    # conv2d: Hout = floor((Hin−kernel_size[0])/stride[0]+1)
    # Maxpool: Hout = floor((Hin-kernel_size[0])/stride[0]+1)
    
    self.conv1 = nn.Conv2d(1, 8, kernel_size=10, stride=4, padding=0, dilation=1)
    # Max pooling of size 2 kernel_size=2, stride=2, padding=0, dilation=1
    T, H = floor((article_max_size - 10)/4 + 1), floor((2*hidden_size - 10)/4 + 1)
    T, H = floor((T - 2)/2 + 1), floor((H - 2)/2 + 1)
    
    self.conv2 = nn.Conv2d(8, 8, kernel_size=7, stride=3)
    # Max pooling of size 2 kernel_size=2, stride=2, padding=0, dilation=1
    T, H = floor((T-7)/3 + 1), floor((H-7)/3 + 1)
    T, H = floor((T-2)/2 + 1), floor((T-2)/2 + 1)
    
    self.conv3 = nn.Conv2d(8, 8, kernel_size=5, stride=2)
    # Max pooling of size 2 kernel_size=2, stride=2, padding=0, dilation=1
    T, H = floor((T-5)/2 + 1), floor((H-5)/2 + 1)
    # T, H = floor((T-2)/2 + 1), floor((T-2)/2 + 1)  # enlevé car pass assez de dim
    print(2*hidden_size, article_max_size)
    print("size reward", 16*T*H, 50)
    self.out_layer1 = nn.Linear(8 * T * H-8, 50)
    self.out_layer2 = nn.Linear(50, 1)
    
    self.indextoword = indextoword
    self.wordtoindex = wordtoindex
    self.SOS_token = SOS_token
    self.EOS_token = EOS_token
    self.PAD_token = PAD_token
    self.batch_size = batch_size
    self.article_max_size = article_max_size

    if USE_CUDA:
        self.encoder.cuda()
        self.conv1.cuda()
        self.conv2.cuda()
        self.out_layer1.cuda()
        self.out_layer2.cuda()
  
  def forward(self, input_seq_ref, input_seq_ref_length, input_seq, input_seq_length):
    """
    Evaluate the reward function
    :param input_seq_ref: ref sequence (article)
    :param input_seq: summary sequence
    """
    # Run through encoder
    # print(input_seq_ref, input_seq_ref_length)
    input_seq = torch.squeeze(input_seq, 1)
    input_seq_ref = torch.squeeze(input_seq_ref, 1)
    #input_seq_ref = torch.transpose(input_seq_ref, 0, 1)
    #input_seq = torch.transpose(input_seq, 0, 1)

    # print(input_seq_ref)
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_seq_length, None)
    encoder_ref_outputs, encoder_ref_hidden = self.encoder(input_seq_ref, input_seq_ref_length, None)
    input_metric = F.pad(encoder_outputs, 
                         (0, 0, 0, 0, 0, self.article_max_size - encoder_outputs.size(0)), 
                         "constant", self.PAD_token)
    input_metric_ref = F.pad(encoder_ref_outputs, 
                         (0, 0, 0, 0, 0, self.article_max_size - encoder_ref_outputs.size(0)), 
                         "constant", self.PAD_token)
    input_metric = torch.transpose(input_metric, 0, 1)
    input_metric_ref = torch.transpose(input_metric_ref, 0, 1)
    print("sortie", input_metric)
    compare_vector = torch.cat((input_metric, input_metric_ref), dim=1)
    compare_vector = torch.unsqueeze(compare_vector, 1)  # Add channel dimension
    print("avant")
    print(compare_vector.size())
    compare_vector = F.max_pool2d(F.relu(self.conv1(compare_vector)), 2)
    print("après C1")
    print(compare_vector.size())
    compare_vector = F.max_pool2d(F.relu(self.conv2(compare_vector)), 2)
    print("après C2")
    print(compare_vector.size())
    # compare_vector = F.max_pool2d(F.relu(self.conv3(compare_vector)), 2)
    compare_vector = F.relu(self.conv3(compare_vector))  # Pas de pooling ici, pas assez de dimension apparement
    print("après C3")
    print(compare_vector.size())
    compare_vector = compare_vector.view(self.batch_size, 1, -1)
    print("maitenant")
    print(compare_vector.size())
    compare_vector = self.out_layer1(compare_vector)
    return torch.squeeze(self.out_layer2(compare_vector))
