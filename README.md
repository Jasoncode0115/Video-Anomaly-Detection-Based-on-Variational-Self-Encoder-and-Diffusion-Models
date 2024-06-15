# Video-Anomaly-Detection-Based-on-Variational-Self-Encoder-and-Diffusion-Models
![image](https://github.com/Jasoncode0115/Video-Anomaly-Detection-Based-on-Variational-Self-Encoder-and-Diffusion-Models/assets/145987720/53854f02-ec8b-41c6-a436-69c3b75d820b)
This is the implementation of my work.  
This is a project assignment used in Professor Liang's computer vision course~
## Dependencies
Python 3.6  
Numpy 1.14.5  
opencv 3.4.2  
pillow 8.4.0  
scikit-learn 0.24.2  
scipy 1.5.4  
torch   1.9.1+cu111  
matplotlib 3.1.1  
## Datasets
USCD Ped2 http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm  
Download the datasets into dataset folder, like ./dataset/ped2/  
## Training
Now you can implemnet the codes based on  reconstruction method.  
The codes are basically based on the reconstruction method, and you can easily implement this as
> python Train.py # for training
## Evaluation
Test your own model
Check your dataset_type (ped2)
> python Evaluate.py # for Evaluation
