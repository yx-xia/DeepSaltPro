# DeepSaltPro: Enhancing Halophilic Protein Prediction Accuracy and Efficiency via Multi-Protein Language Model Integration
# 1.Introduction
We propose a novel computational framework, DeepSaltPro, that integrates pre-trained PLMs and advanced deep learning architectures. Leveraging embeddings from Ankh and ESM-2 pre-trained models, DeepSaltPro automatically extracts critical features from protein sequences without reliance on manual feature engineering. On the other hand, the deep neural network architecture was specifically designed to capture complex structural and sequential patterns by incorporating convolutional neural network (CNN), bidirectional gated recurrent unit (BiGRU), and Kolmogorov-Arnold Network (KAN). The validation results across multiple benchmark datasets show that DeepSaltPro consistently outperforms current state-of-the-art methods, providing a reliable and effective solution for halophilic protein prediction.
# 2.Requirements
```
numpy==1.23.5
pandas==2.2.3
scikit_learn==1.0.2
torch==2.7.0
torchvision==0.22.0
```
# 3.Usage
- run ```Code/get_Ankh_basic.py``` to generate pre-trained feature files.  
- run ```Code/get_ESM-2b_basic.py``` to generate pre-trained feature files.  
- run ```Code/independent_test_result.py ```to obtain the results of the 5-fold CV and independent test set of the model.  
