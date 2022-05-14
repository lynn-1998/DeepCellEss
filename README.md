# DeepCellEss

An interpretable deep learning-based cell line-specific essential protein prediction model. 

The DeepCellEss web server for prediction and visualization available at [http://bioinformatics.csu.edu.cn/DeepCellEss/](http://bioinformatics.csu.edu.cn/DeepCellEss/)


## Requirements

- python=3.7.0
- numpy=1.19.2
- pandas=1.1.5
- scikit-learn=0.24.2
- scipy=1.7.1
- pytorch=1.9.0

## Usage

An demo to train DeepEssCell on the dataset of HCT-116 cell line using linux-64 platform.
#### 1. Clone the repo


    $ git clone https://github.com/lynn-1998/DeepCellEss.git

#### 2. Create and activate the environment


    $ conda create --name deepcelless --file requirments.txt
	$ conda activate deepcelless


#### 3. Train model
The trained models will be saved at file folder '../protein/saved_model/HCT-116/'.


    $ cd code
	$ python main.py protein --cell_line=HCT-116 --gpu=0


#### 4. Specify hyperparameters	


	$ python main.py protein --cell_line=HCT-116 --gpu=0


## License
This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE) file for details


## Concat

Please feel free to contact us for any further questions.
 - Yiming Li lynncsu@csu.edu.cn
 - Min Li limin@mail.csu.edu.cn  
  
  
  
  
