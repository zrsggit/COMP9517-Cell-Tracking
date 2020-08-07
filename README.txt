1.Requirements:	
  Keras 2.4.2
  scipy 1.4.1
  scikit-image 0.17.2
  tensorflow 2.3.0

2.Dictionary Tree:
  ├─DIC-C2DH-HeLa
  │  ├─Sequence 1
  │  ├─Sequence 1 Masks
  │  ├─Sequence 2
  │  ├─Sequence 2 Masks
  │  ├─Sequence 3
  │  └─Sequence 4
  └─DIC_Segmentation
      │  models.py
      │  predict_dataset.py
      │  
      └─DIC-C2DH-HeLa
              unet_model240_nord_s12_0911_PREDICT.h5

3.Run python3 predict_dataset.py and the result will be produced in:
  Sequence 1_RES,
  Sequence 2_RES,
  Sequence 3_RES,
  Sequence 4_RES