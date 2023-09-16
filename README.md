# LungEffnet-Lung_Cancer_Classification
Lung-EffNet: Lung cancer classification using EfficientNet from CT-scan images


# Lung Cancer

Lung cancer (LC) remains a leading cause of death worldwide. Early diagnosis is critical to protect innocent human lives. Computed tomography (CT) scans are one of the primary imaging modalities for lung cancer diagnosis. However, manual CT scan analysis is time-consuming and prone to errors/not accurate. Considering these shortcomings, computational methods especially machine learning and deep learning algorithms are leveraged as an alternative to accelerate the accurate detection of CT scans as cancerous, and non-cancerous. In the present article, we proposed a novel transfer learning-based predictor called, Lung-EffNet for lung cancer classification. Lung-EffNet is built based on the architecture of EfficientNet and further modified by adding top layers in the classification head of the model. Lung-EffNet is evaluated by utilizing five variants of EfficientNet i.e., B0–B4. The experiments are conducted on the benchmark dataset “IQ-OTH/NCCD” for lung cancer patients grouped as benign, malignant, or normal based on the presence or absence of lung cancer. The class imbalance issue was handled through multiple data augmentation methods to overcome the biases. The developed model Lung-EffNet attained 99.10% of accuracy and a score of 0.97 to 0.99 of ROC on the test set. We compared the efficacy of the proposed fine-tuned pre-trained EfficientNet with other pre-trained CNN architectures. The predicted outcomes demonstrate that EfficientNetB1 based Lung-EffNet outperforms other CNNs in terms of both accuracy and efficiency. Moreover, it is faster and requires fewer parameters to train than other CNN based models, making it a good choice for large-scale deployment in clinical settings and a promising tool for automated lung cancer diagnosis from CT scan images. 


## Transfer Learning

In contrast to conventional machine learning methods, CNN enables the automatic extraction of both low-level and high-level feature maps from the model's convolutional base, pooling, and batch-normalization layers. The one-dimensional feature vector created from these extracted feature maps is then sent to a set of single or multiple fully connected layers for classification. Despite its enormous success, one of CNN's drawbacks is that it needs a lot of data samples to train the model effectively and avoid high-bias (underfitting) and high-variance (overfitting) issues. However, it is not practical to gather a significant amount of annotated data for various research challenges, particularly in the field of medical imaging. Additionally, most of the data are not even freely available. To overcome the aforementioned issue, the transfer-learning technique can be applied. Transfer-learning transfers the knowledge taken from architectures that were originally trained on a bigger benchmark dataset such as ImageNet [43] to problems that are either similar to or different from their original context, such as the classification of lung cancer from CT-scan slices with fewer data points. Figure 5 illustrates the broad concept of transfer learning. None of the pre-trained CNN designs can be utilized directly for inference and expect sufficient generalizability on unseen test instances due to the difference in the domains of the source and target dataset, i.e., CT scans. Instead, to adapt to the images in the target domain, the layers of the pre-trained models are refined empirically. Instead, to adapt to the images in the target domain, the layers of the pre-trained models are fine empirically. The technique of fine-tuning involves retraining the weights taken from a few top layers of a deep CNN architecture for different specific problems. These weights were initially trained on a very large dataset. By unfreezing all or some of the layers in convolutional base layers [43], or by employing pre-trained architectures as fixed feature extractors that are then fed to other classifiers like SVM for classification, pre-trained architectures can be fine-tuned.
In this study, transfer learning of five variants of pre-trained EfficientNet i.e., EfficientNet B0 – B4 is performed where each variant of the EfficientNet model is fine-tuned explicitly on CT-scan slices of lung cancer. The feature maps from the EfficientNet are extracted that are then passed to the fully connected layers for classification. The following section goes over the details for optimizing the classification layers of the pre-trained EfficientNet architecture. 



## License 


Please see the License File for more detail.


## Citation

If you find the papers and code helpful for your research, please cite the following paper:

Raza, Rehan, et al. "Lung-EffNet: Lung Cancer Classification Using EfficientNet from CT-scan Images." Engineering Applications of Artificial Intelligence, vol. 126, 2023, p. 106902,  https://doi.org/10.1016/j.engappai.2023.106902. Accessed 16 Sept. 2023.



## Contact

rehanrazag@gmail.com
fati.zulfiar@gmail.com
