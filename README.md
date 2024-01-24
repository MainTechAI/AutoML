# AutoML
The project was made in 2020.
### Installation
```
git clone https://github.com/MainTechAI/AutoML.git
cd AutoML
conda env create --name NEW_ENV_NAME --file=requirements.yml
```

### How to run an experiment:
* Upload your dataset and column descriptions file
* Select desired ML algorithms (e.g., SVM and XGBoost) in settings 
* Select desired metric (e.g., F1-score) in settings 
* Specify the time and size limits of the models
* Click 'Start'

![1](https://user-images.githubusercontent.com/52529117/196767944-3cf55608-d093-4b75-bccd-8d2ed4d20398.png) ![2](https://user-images.githubusercontent.com/52529117/196767946-f5ef7d41-5e00-4878-8639-af99a369ac6e.png)

The application will automatically select the optimal hyperparameters to maximize the selected metric. 
The best N models will be serialized in pickle format and saved in the results folder.

![3](https://user-images.githubusercontent.com/52529117/196767947-fa5e26ec-e7dc-4377-b429-09d52c7cf895.png)


More information [pdf](http://omega.sp.susu.ru/publications/bachelorthesis/2019_403_shchukinma.pdf), [slides](http://omega.sp.susu.ru/publications/bachelorthesis/2019_403_shchukinma_slides.pdf) (in Russian)

___
### Special assignment
During my studies at LUT, I extended the functionality of this project by adding:
* A meta-learning algorithm based on a technique similar to Dataset2Vec. The algorithm populates the 
hyperparameter space of [TPE](https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf) 
with the most promising hyperparameters. 
* Ensemble learning algorithms, namely bagging and voting.
* Imbalanced data handling algorithms that perform over-sampling and/or under-sampling.

Training data for the meta-learning algorithm consists of 512 datasets obtained
from [OpenML](https://www.openml.org/) project. Each dataset is described by 85 meta-features. 

The new functionality is not present in the GUI. 
You need to specify key word arguments of the class _ModelSelection_.
Examples can be found [here](https://github.com/MainTechAI/AutoML/tree/master/auto_ml/examples).
___
TODO:
* Add type hints
* Add docstrings
