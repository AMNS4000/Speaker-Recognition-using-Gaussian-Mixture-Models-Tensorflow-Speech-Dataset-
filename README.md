1.)Made a Speaker Recognition model using 30 different GMMs for 30 speakers

2.)Tensorflow Speech Recognition Dataset utilized.

3.)Used classical ML techniques and data augmentation to achieve an accuracy of 85%

4.)Features like MFCCs, deltas and spectral centroids utilized.

Some Observations and Results :-)

1.) The input audio remains the same at 8Hz, 16Hz, and 22Hz which is the original sampling rate:

Original at 22Hz:
![image](https://github.com/AMNS4000/Speaker-Recognition-using-Gaussian-Mixture-Models-Tensorflow-Speech-Dataset-/assets/104384727/d5448440-e22e-400b-bd4c-0bf8ee09be83)


Now at 16Hz:
![image](https://github.com/AMNS4000/Speaker-Recognition-using-Gaussian-Mixture-Models-Tensorflow-Speech-Dataset-/assets/104384727/47d88589-9bf2-4176-8e78-f39fa64cb452)

2.) By adding Backgroumd Noise and doing Data Augmentation increases the Accuracy

![image](https://github.com/AMNS4000/Speaker-Recognition-using-Gaussian-Mixture-Models-Tensorflow-Speech-Dataset-/assets/104384727/3d837dea-ae23-4467-ab12-411fb252f147)

3.) Features extracted are MFCC's, deltas, double deltas and spectral centroids

![image](https://github.com/AMNS4000/Speaker-Recognition-using-Gaussian-Mixture-Models-Tensorflow-Speech-Dataset-/assets/104384727/8b4a51df-e95c-449c-a3fa-808acdfab12e)
