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
