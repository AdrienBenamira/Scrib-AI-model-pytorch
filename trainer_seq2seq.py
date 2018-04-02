class Seq2SeqTrainer(nn.Module):
  """
  Model used for the human preferences metric
  """
  def __init__(self, attn_model, hidden_size, embed_size, n_layers, wordtoindex, indextoword,
               n_epochs, encoder_temporal,decoder_attention_bol, pairs_train,
                pointeur_bolean, attention_bolean, tie_weights_bolean, num_layer_encoder=1,
                num_layer_decoder=1,dropout=0.5, SOS_token=1, EOS_token=2, PAD_token=0, batch_size=5,
                clip=50,learning_rate = 0.0001, article_max_size=500, decoder_learning_ratio=5,
                plot_every=20, print_every = 100, evaluate_every = 1000, max_length=100):
    super(Seq2SeqTrainer, self).__init__()
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
    # Configure models
    self.attn_model = attn_model
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.dropout = dropout
    self.batch_size = batch_size
    self.embed_size=embed_size
    # Configure training/optimization
    self.clip = clip
    self.learning_rate = learning_rate
    self.decoder_learning_ratio = decoder_learning_ratio
    self.n_epochs = n_epochs
    self.plot_every = plot_every
    self.print_every = print_every
    self.evaluate_every = evaluate_every
    #config type model
    self.encoder_temporal=encoder_temporal
    self.decoder_attention_bol=decoder_attention_bol
    self.pointeur_bolean=pointeur_bolean
    self.attention_bolean=attention_bolean
    self.tie_weights_bolean=tie_weights_bolean
    self.indextoword=indextoword
    self.wordtoindex=wordtoindex
    self.pairs_train=pairs_train
    # Initialize models
    self.encoder = EncoderRNN(len(indextoword), embed_size, hidden_size,pretrained_weight, n_layers=n_layers, dropout=dropout)
    self.decoder = DecoderStep(hidden_size, embed_size,len(indextoword), n_layers, dropout_p=dropout, temporal=encoder_temporal,de_att_bol=decoder_attention_bol,point_bol=pointeur_bolean,attention_bol=attention_bolean )
    self.max_length=max_length
    # Move models to GPU
    if USE_CUDA:
        self.encoder.cuda()
        self.decoder.cuda()


  def train_step(self, input_batches, input_lengths, target_batches, target_lengths, encoder_optimizer, decoder_optimizer, criterion, target_context, target_pointer,E_hist=None):
    """
      
      
      
    """
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    total_loss = 0 # Added onto for each word
    # Run words through encoder
    encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)
    # Share embedding
    self.decoder.embedding.weight = self.encoder.embedding.weight
    # Optionally tie weights as in:
    # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
    if self.tie_weights_bolean:
      # Depend of the configuration you choose, the dimmensions must be : E= H or 2*H ou 3*H
      decoder.out.weight = encoder.embedding.weight
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:self.decoder.n_layers] # Use last (forward) hidden state from encoder
    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, self.batch_size, self.decoder.output_size))
    all_decoder_outputs_context = Variable(torch.zeros(max_target_length, self.batch_size, input_lengths[0]))
    all_decoder_outputs_pointer = Variable(torch.zeros(max_target_length, self.batch_size,1))
    hidden_history_decoder = decoder_hidden[-1].unsqueeze(0)
    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
        all_decoder_outputs_context=all_decoder_outputs_context.cuda()
        all_decoder_outputs_pointer=all_decoder_outputs_pointer.cuda()
    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn, E_hist,hidden_history_decoder = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs,E_hist,t, hidden_history_decoder, input_batches)
        if self.pointeur_bolean:
          all_decoder_outputs[t] = decoder_output[0]
          all_decoder_outputs_context[t] = decoder_output[1]
          all_decoder_outputs_pointer[t] = decoder_output[2]
        else:        
          all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target
    # Loss calculation and backpropagation
    if self.pointeur_bolean:
      #1er loss : le pointeur
      all_decoder_outputs_pointer=all_decoder_outputs_pointer.squeeze(2)
      loss1=criterion(all_decoder_outputs_pointer,torch.max(target_pointer, 1)[1])
      if 1 in target_pointer.cpu().data.numpy():
        #update tous les poids
        loss2 = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths)
        total_loss= sum([loss1,loss2])
      else:
        loss2 = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths) 
        #loss sur l'attention en pointer, ca ne va pas tuer l'attention ?
        #loss3 = 
        total_loss= sum([loss1,loss2])
    else:    
      total_loss = masked_cross_entropy(
          all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
          target_batches.transpose(0, 1).contiguous(), # -> batch x seq
          target_lengths)
    total_loss.backward()
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
    dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)
    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return total_loss.data[0], ec, dc
        
      
  def evaluate(self, input_seq):
    """
    
    
    
    """
    input_lengths = [len(input_seq.split())]
    input_seqs = [indexes_from_sentence(wordtoindex, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
    words_input=input_seq.split(' ')
    if USE_CUDA:
        input_batches = input_batches.cuda()
    # Set to not-training mode to disable dropout
    self.encoder.train(False)
    self.decoder.train(False)
    # Run through encoder
    encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)
    # Share embedding
    self.decoder.embedding.weight = self.encoder.embedding.weight
    # Optionally tie weights as in:
    # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
    if self.tie_weights_bolean:
      # Depend of the configuration you choose, the dimmensions must be : E= H or 2*H ou 3*H
      self.decoder.out.weight = self.encoder.embedding.weight
    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True) # SOS
    decoder_hidden = encoder_hidden[:self.decoder.n_layers] # Use last (forward) hidden state from encoder
    hidden_history_decoder = decoder_hidden[-1].unsqueeze(0)
    E_hist = None
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
    # Store output words and attention states
    decoded_words = []
    
    decoder_attentions = torch.zeros(max(input_lengths) + 1, max(input_lengths) + 1)
    # Run through decoder
    for di in range(self.max_length):
        decoder_output, decoder_hidden, decoder_attention, E_hist,hidden_history_decoder = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs,E_hist,di, hidden_history_decoder,input_batches
        )
        if self.attention_bolean:
          if self.pointeur_bolean:
            [decoder_output_voc, decoder_output_context, decoder_output_pointer]=decoder_output
            decoder_attentions[di,:decoder_attention.size(2)] += decoder_output_context.squeeze(0).cpu().data
        if self.pointeur_bolean:
          [decoder_output_voc, decoder_output_context, decoder_output_pointer]=decoder_output
          if decoder_output_pointer.cpu().data.numpy()[0][0]>0.5:
            topv, topi = decoder_output_context.data.topk(1)
            ni = topi[0][0]
            topv2, topi2 = decoder_output_context.data.topk(2)
            ni2 = topi2[0][0]
            if len(decoded_words)>=2:
              if decoded_words[-1]==decoded_words[-2]:
                if words_input[ni]==decoded_words[-1]:
                  ni=ni2
            decoded_words.append(words_input[ni])
            try:
              ni=wordtoindex[words_input[ni]]
            except Exception:
              ni=3
            decoder_input = Variable(torch.LongTensor([ni]))
          else:
            topv, topi = decoder_output_voc.data.topk(1)
            ni = topi[0][0]
            topv2, topi2 = decoder_output_context.data.topk(2)
            ni2 = topi2[0][0]
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
              if len(decoded_words)>=2:
                if decoded_words[-1]==decoded_words[-2]:
                  if indextoword[ni]==decoded_words[-1]:
                    ni=ni2              
              decoded_words.append(self.indextoword[ni])
              decoder_input = Variable(torch.LongTensor([ni]))
        else:        
          # Choose top word from output
          topv, topi = decoder_output.data.topk(1)
          ni = topi[0][0]
          topv2, topi2 = decoder_output.data.topk(2)
          ni2 = topi2[0][0]
          if ni == EOS_token:
              decoded_words.append('<EOS>')
              break
          else:
              if len(decoded_words)>=2:
                if decoded_words[-1]==decoded_words[-2]:
                  if indextoword[ni]==decoded_words[-1]:
                    ni=ni2
              decoded_words.append(self.indextoword[ni])
              decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA:
          decoder_input = decoder_input.cuda()
    # Set back to training mode
    self.encoder.train(True)
    self.decoder.train(True)

    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]
  
  def evaluate_and_show_attention(self, input_sentence, target_sentence=None):
    output_words, attentions = self.evaluate(input_sentence)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)
    
    show_attention(input_sentence, output_words, attentions)
  
  
  def evaluate_randomly(self):
    [input_sentence, target_sentence] = random.choice(pairs_train)
    self.evaluate_and_show_attention(input_sentence, target_sentence)
    
  
  
  def train_all(self):
    epoch=0
    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
    decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate * self.decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()
    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every
    # Begin!
    ecs = []
    dcs = []
    eca = 0
    dca = 0
    while epoch < self.n_epochs:
        print("Epoch:", epoch)
        epoch += 1
        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths,target_context, target_pointer,_ = random_batch(self.batch_size,self.pairs_train, self.wordtoindex)
        # Run the train function
        loss, ec, dc = self.train_step(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder_optimizer, decoder_optimizer, criterion,target_context, target_pointer
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
            ecs.append(eca / plot_every)
            dcs.append(dca / plot_every)
            eca = 0
            dca = 0
            plot_loss_total = 0
            
            
  def train_master_piece(self, input_batches, input_lengths, encoder_optimizer, decoder_optimizer,input_letters,reward_predictor):
    

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    total_loss = 0 # Added onto for each word

          
    # Run words through encoder
    if torch.cuda.is_available():
        input_batches = input_batches.cuda()
    encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)
    
    # Share embedding
    self.decoder.embedding.weight = self.encoder.embedding.weight
    
    # Optionally tie weights as in:
    # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
    if tie_weights_bolean:
      # Depend of the configuration you choose, the dimmensions must be : E= H or 2*H ou 3*H
      self.decoder.out.weight = self.encoder.embedding.weight
    
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:self.decoder.n_layers] # Use last (forward) hidden state from encoder
    E_hist = None
    decoded_words = [[] for y in range(batch_size)]
    max_target_length = max(target_lengths)
    hidden_history_decoder = decoder_hidden[-1].unsqueeze(0)
    
    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
    # Run through decoder one time step at a time
    
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn, E_hist,hidden_history_decoder = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs,E_hist,t, hidden_history_decoder, input_batches
        )
        

          
        #REMPLACER ni par [ni]*Batch (pour chaque batch déterminer ni)
        
        if pointeur_bolean:
          [decoder_output_voc, decoder_output_context, all_decoder_outputs_pointer ]=decoder_output
          
          if all_decoder_outputs_pointer.cpu().data.numpy()[0][0]>0.5:
            topv, topi = decoder_output_context.data.topk(1)
            decoder_input = Variable(topi.squeeze(1))
            for batch, val  in enumerate(decoder_input):
              try:
                decoded_words[batch].append(input_letters[batch][val.cpu().data[0]])
              except Exception:
                 decoded_words[batch].append('UNK')
              

         #cest chaud
          else:
            topv, topi = decoder_output_voc.data.topk(1)
            decoder_input = Variable(topi.squeeze(1))
            for batch, val  in enumerate(decoder_input):
              decoded_words[batch].append(indextoword[val.cpu().data[0]])
         
        else:        
          # Choose top word from output
          topv, topi = decoder_output.data.topk(1)
          #decoded_words.append(indextoword[ni])
          decoder_input = Variable(topi.squeeze(1))
          for batch, val  in enumerate(decoder_input):
            decoded_words[batch].append(indextoword[val.cpu().data[0]])
        if USE_CUDA:
          decoder_input = decoder_input.cuda()
    final_in=[]
    final_out=[]
    for batch, val  in enumerate(decoder_input):
      final_in.append(' '.join(input_letters[batch]))
      final_out.append(' '.join(decoded_words[batch]))  
    total_loss=reward_predictor.eval(final_in, final_out)
    total_loss=total_loss.sum()
    total_loss.backward()
      
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return total_loss.data[0], ec, dc
