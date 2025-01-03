## Generating Text with PyTorch:

### 1. Building Bigram Models: 

**Vocabulary Size (Chapter 1)**: 321

**Number of bigrams in Chapter 1**: 854

**First 5 bigrams**: 
 [[131 130]
 [130   0]
 [  0 284]
 [284 289]
 [289   4]
 [  4 264]
 [264   0]
 [  0 244]
 [244 156]
 [156 124]]         

**First 10 features:** tensor([131, 130,   0, 284, 289,   4, 264,   0, 244, 156])

**First 10 labels:** tensor([130,   0, 284, 289,   4, 264,   0, 244, 156, 124])      

          
### 2. Build Sequential Model Class       

**NextWordBigram Sequential Model:**      

NextWordBigram(

  (embedding): Embedding(321, 2)
  
  (linear1): Linear(in_features=2, out_features=18, bias=True)
  
  (linear2): Linear(in_features=18, out_features=321, bias=True)

)   
  
**Generated Text:**

"it is a truth bit high so bit high so bit high so bit" 

The text generated from the untrained model seems to be repeating the tokens 'hope ', and 'impatiently'. The first context token 'truth' is predicted to have the next token 'hope'. The previously predicted token 'hope' becomes the next context token which is used to predict the next token 'impatiently'. The previously predicted token 'impatiently' becomes the next context token which is used to predict the next token 'hope'. This causes the repeated predictions 'hope' ==> 'impatiently' ==> 'hope ' ==> 'impatiently'.

### 3. Training Bigram Model
![ALT TEXT](https://github.com/SaifurRR/NLP-Large-Language-Models/blob/main/Generating%20Text%20with%20PyTorch/epoch_prob.jpg)

Cross Entropy Loss (CELoss) for number of epochs = 700. It looks like the trained bigram model generates slightly more coherent text than the untrained bigram model. Specifically, it correctly predicted the next three tokens: 'universally', 'acknowledged', and 'that'. However, we see the limitations of the bigram model as it quickly falls into repetition, repeating the sequence "that he is not you must know mrs". Since bigram models only use a single context token (the previous token) to predict the next token, it won't be able to learn the long-term contextual relationships within the text beyond those immediate bigram pairs.


### 4. Preprocessing Text for RNNs

**Vocabulary size:** 51

### 5. LSTM Model

CharacterLSTM(

  (embedding): Embedding(51, 64)

  (lstm): LSTM(64, 128, batch_first=True)
  
  (linear): Linear(in_features=128, out_features=51, bias=True)

)

### 6. Training a LSTM Model

![ALT TEXT](https://github.com/SaifurRR/NLP-Large-Language-Models/blob/main/Generating%20Text%20with%20PyTorch/epoch_LSTM.jpeg)

The LSTM model was able to successfully generate the full first sentence: **"It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife"**. 

Notably, it is even able to correctly generate the comma ',' after the word 'acknowledged'. This is because we decided to not remove punctuations and special characters from the raw text which allowed the LSTM to learn to generate them instead. Afterward, it starts to deviate from the actual text but still maintains some grammatical accuracy. Note that, our model was trained fairly shortly and only on a small portion of the full text.


### Generating Text with Frankenstein Using PyTorch

The first letter in Frankenstein contains **6,850** character-based tokens, with a vocabulary of **60** unique tokens, including special characters, numbers, uppercase, and lowercase letters.

**LSTM Model:**

CharacterLSTM(

  (embedding): Embedding(60, 48)
  
  (lstm): LSTM(48, 96, batch_first=True)
  
  (linear): Linear(in_features=96, out_features=60, bias=True)

)

**Optimizer:**

Adam (

Parameter Group 0

    amsgrad: False
    
    betas: (0.9, 0.999)
    
    capturable: False
    
    differentiable: False
    
    eps: 1e-08
    
    foreach: None
    
    fused: None
    
    lr: 0.015
    
    maximize: False
    
    weight_decay: 0

)

**Cross Entropy Loss:**

Epoch [1/5], CELoss: 1.0588

Epoch [2/5], CELoss: 0.6542

Epoch [3/5], CELoss: 0.4657

Epoch [4/5], CELoss: 0.3593

Epoch [5/5], CELoss: 0.3722

### Generated Text:

**Starting Prompt:** "You will rejoice to hear"

**Generated Text:** "You will rejoice to hear that no disaster has accompanied the commencement of an enterprise which attracts the needle and may regulate a thousand celestial observations that require only this voyage to render their seeming eccentricities consistent for ever. I shall satiate my ardent curiosity with the sight of a part often depressed. I am about to proceed on a long and difficult voyage, the emergencies of which will demand all my fortitude: I am required not only to raise the second dignity in the vessel and entreated"

