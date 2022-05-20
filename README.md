# [<img style="height:40px" src="http://bioinformatics.csu.edu.cn/DeepCellEss/static/imgs/Logo.svg">](http://bioinformatics.csu.edu.cn/DeepCellEss)  DeepCellEss 

An interpretable deep learning-based cell line-specific essential protein prediction model. 

The DeepCellEss web server for prediction and visualization available at [http://bioinformatics.csu.edu.cn/DeepCellEss](http://bioinformatics.csu.edu.cn/DeepCellEss)


## Requirements

- python=3.7.0
- numpy=1.19.2
- pandas=1.1.5
- scikit-learn=0.24.2
- scipy=1.7.1
- pytorch=1.9.0
- gensim=3.8.3

## Usage

An demo to train DeepEssCell on the dataset of HCT-116 cell line using linux-64 platform.
#### 1. Clone the repo


<<<<<<< HEAD
    git clone https://github.com/lynn-1998/DeepCellEss.git
=======
    $ git clone https://github.com/lynn-1998/DeepCellEss.git
    $ cd DeepCellEss
>>>>>>> a73f55f623b05a2eb4359155f60f2766c5fe6564

#### 2. Create and activate the environment


<<<<<<< HEAD
    cd DeepCellEss
	conda create --name deepcelless --file requirments.txt
	conda activate deepcelless
=======
    $ conda create --name deepcelless --file requirments.txt
 	$ conda activate deepcelless
>>>>>>> a73f55f623b05a2eb4359155f60f2766c5fe6564


#### 3. Train model
The trained models will be saved at file folder '../protein/saved_model/HCT-116/'.


    cd code
	python main.py protein --cell_line HCT-116 --gpu 0


#### 4. Specify model hyperparameters	

>***--cell_line*** is the name of cell line benchmark dataset to train.  
>***--kernel_size*** is the kernel number of the CNN layer.  
>***--head_num*** is the number of attention heads.  
>***--hidden_dim*** is the dimention of the hidden state vector.  
>***--layer_num*** is the number of lstm layers.  
>***--gpu*** is the gpu number you used to build and train the model. The defalt value of 0 means "cuda:0". No gpu will default to cpu.


## License
This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE) file for details


## Concat

Please feel free to contact us for any further questions.
 - Yiming Li lynncsu@csu.edu.cn
 - Min Li limin@mail.csu.edu.cn  
  
  
  
  
