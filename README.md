## Kaggle quora insincere questions competition

----
https://www.kaggle.com/c/quora-insincere-questions-classification/

Embeddings: Glove + Paragram averaging 
  
Architecture: meta-embedding -- LSTM -- GRU -- concat(maxpool, avgpool) -- sigmoid  
Training/validation: one-cycle policy ([ref paper](https://arxiv.org/pdf/1803.09820.pdf)) with 5-fold cross-validation (5 epochs for each)  
  
Conclusion:  
The single model performence acquired from above simple setup is pretty well (~ top 1%). The key for such result could be the one-cycle policy for training the model. The strategy can allow us to use large learning rate with few epochs, which bring us model regularization and short convergence time. Besides, the cv score is more robust for model validation than the public LB. (the train set is 10 times larger than public test set)

### Dependencies
Pytorch 1.0.1
