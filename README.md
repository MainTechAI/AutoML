# AutoML

How to run an experiment:
* Upload your dataset and column descriptions file
* Select desired ML algorithms (e.g., SVM and XGBoost) in settings 
* Select desired metric (e.g., F1-score) in settings 
* Specify the time and size limits of the models
* Click 'Start'

The application will automatically select the optimal hyperparameters to maximize the selected metric. 
The best N models will be serialized in pickle format and saved in the results folder.


More inforamtion [pdf](http://omega.sp.susu.ru/publications/bachelorthesis/2019_403_shchukinma.pdf), [slides](http://omega.sp.susu.ru/publications/bachelorthesis/2019_403_shchukinma_slides.pdf) (in Russian)

_____

API updates is a top priority (GUI last).

(TODO) Describe what the application is intended for. Add screenshots.
