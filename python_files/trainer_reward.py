import torch
from modele_reward import RewardPredictor
from modele_seq2seq import EncoderRNN, DecoderStep
from utils_loss_visualization import masked_cross_entropy

USE_CUDA =torch.cuda.is_available()
print('On utilise le GPU, '+str(USE_CUDA)+' story')
import random
import time
import torchtext.vocab as vocab
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim


USE_CUDA =torch.cuda.is_available()
print('On utilise le GPU, '+str(USE_CUDA)+' story')

#HYPERPARAMETRES TEST
MIN_LENGTH_ARTICLE=10
MIN_LENGTH_SUMMARY=5
MAX_LENGTH_OUTPUT_GENERATE=50
dim=100
glove = vocab.GloVe(name='6B', dim=dim)

#TOKENS
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class RewardPredictorTrainer:
    """
    Model used for the human preferences metric
    """
    def __init__(self, input_size, embed_size, hidden_size, output_size, indextoword,
                 wordtoindex, num_layer_encoder=1, num_layer_decoder=1, 
                 dropout=0.5, SOS_token=1, EOS_token=2, PAD_token=0, batch_size=5, article_max_size=500):
        """
        :param input_size:
        :param embed_size:
        :param hidden_size:
        :param output_size:
        :param indextoword:
        :param wordtoindex
        :param num_layer_encoder:
        :param num_layer_decoder:
        :param dropout:
        :param SOS_token:
        :param EOS_token:
        :param PAD_token:
        :param batch_size:
        :param article_max_size:
        """
        self.encoder = EncoderRNN(input_size, embed_size, hidden_size, pretrained_weight, num_layer_encoder, dropout)
        self.decoder = DecoderStep(hidden_size, embed_size, output_size, num_layer_decoder, temporal=False,
                                   de_att_bol=False, point_bol=False, dropout_p=dropout, attention_bol=False)
        self.predictor = RewardPredictor(input_size, embed_size, hidden_size, 
                                         indextoword,wordtoindex, num_layer_encoder, 
                                         dropout, SOS_token, EOS_token, PAD_token, 
                                         batch_size, article_max_size)
        self.indextoword = indextoword
        self.wordtoindex = wordtoindex
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.PAD_token = PAD_token
        self.batch_size = batch_size
        self.article_max_size = article_max_size
        
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()
            self.predictor.cuda()

    def train_step_autoenc(self, input_batches, input_lengths,
              encoder_optimizer, decoder_optimizer, criterion, 
              max_length=1000, E_hist=None):
        """
        Train step for the autoencoder
        """
        # The targets are the inputs
        target_batches = input_batches
        target_lengths = input_lengths
        # Zero gradients of both optimizers
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0 # Added onto for each word

        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)

        # Share embedding
        self.decoder.embedding.weight = self.encoder.embedding.weight

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = encoder_hidden[:self.decoder.n_layers] # Use last (forward) hidden state from encoder

        max_target_length = max(target_lengths)
        all_decoder_outputs = Variable(torch.zeros(max_target_length, self.batch_size, self.decoder.output_size))
        hidden_history_decoder = decoder_hidden[-1].unsqueeze(0)

        # Move new Variables to CUDA
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()

        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn, E_hist,hidden_history_decoder = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs,E_hist,t, hidden_history_decoder, input_batches
            )

            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t] # Next input is current target

        # Loss calculation and backpropagation
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)

        # Update parameters with optimizers
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0], ec, dc
      
    def train_autoenc(self, learning_rate, decoder_learning_ratio, n_epochs, batch_size, 
              pairs_train, wordtoindex):
      """
      Training of the autoencoder
      """
      encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
      decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
      criterion = nn.CrossEntropyLoss()
      # Keep track of time elapsed and running averages
      start = time.time()
      plot_losses = []
      print_loss_total = 0 # Reset every print_every
      plot_loss_total = 0 # Reset every plot_every
      ecs = []
      dcs = []
      eca = 0
      dca = 0
      epoch = 0
      
      while epoch < n_epochs:
          epoch += 1

          print(epoch)

          #print (epoch)
          # Get training data for this cycle
          input_batches, input_lengths, target_batches, target_lengths,target_context, target_pointer,_ = random_batch(batch_size,pairs_train, wordtoindex)
          # Run the train function
          """
          self, input_batches, input_lengths, target_lengths,
              encoder_optimizer, decoder_optimizer, criterion, 
              max_length=1000, E_hist=None):
              """
          loss, ec, dc = self.train_step_autoenc(
              input_batches, input_lengths,
              encoder_optimizer, decoder_optimizer, criterion
          )

          # Keep track of loss
          print_loss_total += loss
          plot_loss_total += loss
          eca += ec
          dca += dc

          if epoch == 5:
              self.evaluate_randomly()

          if epoch % print_every == 0:
              print_loss_avg = print_loss_total / print_every
              print_loss_total = 0
              print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
              print(print_summary)
              show_losses(ecs,dcs,plot_losses)

          if epoch % evaluate_every == 0:
              self.evaluate_randomly()

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
              
    def evaluate_and_show_attention(self, input_sentence, target_sentence=None):
      output_words, attentions = self.eval_autoenc(input_sentence)
      output_sentence = ' '.join(output_words)
      print('>', input_sentence)
      if target_sentence is not None:
          print('=', target_sentence)
      print('<', output_sentence)

      #show_attention(input_sentence, output_words, attentions)
  
  
    def evaluate_randomly(self):
      [input_sentence, target_sentence] = random.choice(pairs_train)
      self.evaluate_and_show_attention(input_sentence, target_sentence)


    def eval_autoenc(self, input_seq, training_mode=False, max_length=50):
        USE_CUDA = torch.cuda.is_available()
        input_lengths = [len(input_seq.split())]
        
        input_seq = indexes_from_sentence(self.wordtoindex, input_seq)
        input_batches = Variable(torch.LongTensor([input_seq]), volatile=True).transpose(0, 1)

        if USE_CUDA:
            input_batches = input_batches.cuda()

        # Set to not-training mode to disable dropout
        if not training_mode:
            self.encoder.train(False)
            self.decoder.train(False)

        # Run through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)

        # Create starting vectors for self.decoder
        decoder_input = Variable(torch.LongTensor([self.SOS_token]), volatile=True)  # SOS
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # Use last (forward) hidden state from self.encoder
        hidden_history_decoder = decoder_hidden[-1].unsqueeze(0)
        E_hist = None
        if USE_CUDA:
            decoder_input = decoder_input.cuda()

        # Store output words and attention states
        decoded_words = []
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

        # Run through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention, E_hist, hidden_history_decoder = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, E_hist, di, hidden_history_decoder, input_batches
            )
            #decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == self.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.indextoword[ni])

            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ni]))
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

        if not training_mode:
            # Set back to training mode
            self.encoder.train(True)
            self.decoder.train(True)
        return decoded_words, encoder_outputs
      
    def train_step_predictor(self, input_batches, input_length, input_ref_batches, input_ref_length):
        """
        Step to train the reward predictor
        """
        USE_CUDA = torch.cuda.is_available()
        
        input_batches = Variable(torch.LongTensor([input_batches])).transpose(0, 1)
        input_batches_ref = Variable(torch.LongTensor([input_ref_batches])).transpose(0, 1)
        
        if USE_CUDA:
            input_batches = input_batches.cuda()
            input_batches_ref = input_batches_ref.cuda()
            
        
        return self.predictor(input_batches_ref, input_ref_length, input_batches, input_length)   
      

    def eval(self, input_seq_ref, input_seq, is_training=False):
        """
        Evaluate the reward function
        :param input_seq_ref: ref sequence
        :param input_seq: summary sequence
        """
        USE_CUDA = torch.cuda.is_available()
        to_indexes_fn = lambda x: indexes_from_sentence(self.wordtoindex, x)
        input_seq = list(map(to_indexes_fn, input_seq))
        
        input_batches_ref = list(map(to_indexes_fn, input_seq_ref))
        input_seq = sorted(input_seq, key=lambda p: len(p), reverse=True)
        input_batches_ref = sorted(input_batches_ref, key=lambda p: len(p), reverse=True)
        
        input_lengths = list(map(len, input_seq))
        input_lengths_ref = list(map(len, input_batches_ref))
        
        input_seq = [pad_seq(s, max(input_lengths)) for s in input_seq]
        input_batches_ref = [pad_seq(s, max(input_lengths_ref)) for s in input_batches_ref]
        
        input_batches = Variable(torch.LongTensor(input_seq)).transpose(0, 1)
        input_batches_ref = Variable(torch.LongTensor(input_batches_ref)).transpose(0, 1)

        if USE_CUDA:
            input_batches = input_batches.cuda()
            input_batches_ref = input_batches_ref.cuda()

        # Set to not-training mode to disable dropout
        if not is_training:
            self.predictor.train(False)
        
        print('input_batches_ref.size()')
        print(input_batches_ref.size())
        print('input_lengths_ref.size()')
        print(input_lengths_ref)
        print('input_batches.size()')
        print(input_batches.size())
        print('input_lengths.size()')
        print(input_lengths)
        loss = self.predictor(input_batches_ref, input_lengths_ref, input_batches, input_lengths)
        
        if not is_training:
            self.predictor.train(True)
          
        return loss
        
    def train(self, input_seq_ref, input_seq, input_seq_ref_2, input_seq_2, proba_density, n_epochs, learning_rate):
        """
        Train the reward predictor
        :param input_seq_ref: article 1
        :param input_seq: summary 1
        :param input_seq_ref_2: article 2
        :param input_seq_2: summary 2
        :param proba_density: list of list [1, 0] if first is better, [0, 1] if second is better, [0.5, 0.5] is equally good
        :param n_epochs:
        :param learning_rate:
        """
        optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)
        
        USE_CUDA = torch.cuda.is_available()
        
        all_input_sequences = input_seq
        all_ref_sequences = input_seq_ref
        all_input_sequences_2 = input_seq_2
        all_ref_sequences_2 = input_seq_ref_2
        all_target_scores = proba_density
        
        epoch = 0
        print_loss_total = 0
        plot_loss_total = 0
        eca = 0
        
        
        while epoch < n_epochs:
            epoch += 1

            print("Epoch:", epoch)

            #print (epoch)
            # Get training data for this cycle
            input_seq, input_length, input_seq_ref, input_ref_length, order_batch1, \
                input_seq_2, input_length_2, input_seq_ref_2, \
                input_ref_length_2, order_batch2, target_scores = random_batch_three(self.batch_size, all_input_sequences, all_ref_sequences, all_input_sequences_2, all_ref_sequences_2,  all_target_scores, self.wordtoindex)
            # Run the train function
            """
            self, input_batches, input_lengths, target_lengths,
              encoder_optimizer, decoder_optimizer, criterion, 
              max_length=1000, E_hist=None):
              """
            
            target_scores = Variable(torch.Tensor(target_scores)).transpose(0, 1)
            P1_2 = Variable(torch.Tensor(self.batch_size))
            P2_1 = Variable(torch.Tensor(self.batch_size))
        
            if USE_CUDA:
                target_scores = target_scores.cuda()
                P1_2 = P1_2.cuda()
                P2_1 = P2_1.cuda()
            
            # Zero gradients of both optimizers
            optimizer.zero_grad()
            
            output1 = self.train_step_predictor(
              input_seq, input_length, input_seq_ref, input_ref_length
            )
            
            output2 = self.train_step_predictor(
              input_seq_2, input_length_2, input_seq_ref_2, input_ref_length_2
            )
            
            # Calculate probas
            for k in range(self.batch_size):
                index_1 = order_batch1.index(k)
                index_2 = order_batch2.index(k)
                P1_2[k] = output1[index_1] / (output1[index_1] + output2[index_2])
                P2_1[k] = output2[index_2] / (output1[index_1] + output2[index_2])
            # Calculate loss
            loss = torch.sum(- target_scores[0] * torch.log(P1_2) - target_scores[1] * torch.log(P2_1))            

            # Loss calculation and backpropagation
            loss.backward()

            # Clip gradient norms
            c = torch.nn.utils.clip_grad_norm(self.predictor.parameters(), clip)

            # Update parameters with optimizers
            optimizer.step()

            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
                print(print_summary)
                show_losses(None,None,plot_losses)

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                plot_loss_total = 0
        
