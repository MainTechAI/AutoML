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

![1](images/1.png)  ![2](images/2.png)

The application will automatically select the optimal hyperparameters to maximize the selected metric. 
The best N models will be serialized in pickle format and saved in the results folder.

![3](images/3.png)


More information [pdf](http://omega.sp.susu.ru/publications/bachelorthesis/2019_403_shchukinma.pdf), [slides](http://omega.sp.susu.ru/publications/bachelorthesis/2019_403_shchukinma_slides.pdf) (in Russian)


_____

- [ ] Fix the data preprocessing function used in the GUI.
- [ ] API updates is a top priority (GUI last)
- [ ] Describe what the application is intended for
- [ ] Add screenshots
- [ ] Data loading. Add other column separators. Not only comma.

