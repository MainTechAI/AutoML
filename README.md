# AutoML

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




