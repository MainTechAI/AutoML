# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog

import os
from os.path import expanduser
import sys
import numpy

from utility.dialog import Ui_Dialog, Ui_WarningPaths, Ui_WarningName, Ui_WarningModels

"""
from .ui to .py  
pyuic5 -o pyfilename.py design.ui
 -x executable  if __name__ == "__main__":
"""


# %%
import time
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread #QRunnable

class TimerThread(QThread):
    
    signal_timer = pyqtSignal(int)
    signal_timer_finish = pyqtSignal()
       
    @pyqtSlot(int)
    def slot_timer_start(self,value):
        self.seconds=value
#        print("Start value:",value)   
    
    def run(self): 
        count = self.seconds
        while count > 0:
            time.sleep(1)
            count -= 1
#            print("seconds remaining",count)
            self.signal_timer.emit(count)
        self.signal_timer_finish.emit()
        
# %%
        
        
        
# %%

import auto
  
class ModelSeletionThread(QThread):
    
    signal_model_selection_finish = pyqtSignal()       
       
    @pyqtSlot(numpy.ndarray,list,str,int,float,int,int,int,dict,str,str,str,int) #!!!
    def slot_start_model_selection_(self,DS,CD,experiment_name,duration,
                                    min_accuracy,max_model_memory,
                                    max_prediction_time,max_train_time,
                                    used_algorithms,metric,validation,
                                    saved_models_count,iterations):
        self.DS=DS
        self.CD=CD
        self.experiment_name=experiment_name
        self.duration=duration
        self.min_accuracy=min_accuracy
        self.max_model_memory=max_model_memory
        self.max_prediction_time=max_prediction_time
        self.max_train_time=max_train_time
        self.used_algorithms=used_algorithms
        self.metric=metric
        self.validation=validation 
        self.saved_models_count=saved_models_count
        self.iterations=iterations

    
    def run(self): 
        # MS=
        auto.ModelSelection(self.DS, self.CD, self.experiment_name, 
                self.duration, self.min_accuracy, self.max_model_memory, 
                self.max_prediction_time, self.max_train_time, 
                self.used_algorithms, self.metric, self.validation, 
                self.saved_models_count,self.iterations)
        
        # private по этому не сможешь вызвать
        #print(MS.__check_time()) 
        
        # публичный метод по этому можешь вызвать
        #MS.fit(5) # может отсюда управлять?
        print("ModelSeletionThread finish")
        
        self.signal_model_selection_finish.emit()
        

# %%


# %%
class Ui_MainWindow(QMainWindow):
    signal_start_timer = pyqtSignal(int)
    signal_start_MS = pyqtSignal(
            numpy.ndarray,list,str,int,float,int,int,int,dict,str,str,str,int)#!!!
    
    def __init__(self):   
        super(Ui_MainWindow, self).__init__()
        
        self.experiment_name = 'experiment_1'
        self.duration = 240
        self.min_accuracy = 65.0
        self.max_model_memory = 1048576
        self.max_prediction_time = 40 
        self.max_train_time = 30
        self.iterations=100
        
        self.dataset_path = None
        self.column_description_path = None
        
        self.used_algorithms = {
        'AdaBoost':True, 'XGBoost':True, 'Bagging(SVС)':True,
        'MLP':True, 'HistGB':False, 'Ridge':False,
        'LinearSVC':False, 'PassiveAggressive':False, 'LogisticRegression':False,
        'LDA':False, 'QDA':False, 'Perceptron':False,      
        'SVM':True, 'RandomForest':True, 'xRandTrees':True,
        'ELM':True, 'DecisionTree':False, 'SGD':False,
        'KNeighbors':False, 'NearestCentroid':False, 'GaussianProcess':False,
        'LabelSpreading':False, 'BernoulliNB':False, 'GaussianNB':False,
        'DBN':False, 'FactorizationMachine':False, 'PolynomialNetwork':False
        }
              
        
        self.metric='accuracy'
        self.validation='3 fold CV'
        self.saved_models_count='Топ 5'
        
        self.DS = None
        self.CD = None
        
        self.lcd_bool = True
        
        self.dialog_settings = Ui_Dialog()
        self.warning_paths = Ui_WarningPaths()
        self.warning_name = Ui_WarningName()
        
        self.warning_models = Ui_WarningModels()
        
        self.setupUi()
        
       
    
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.setEnabled(True)
        self.resize(320, 480)
        self.setWindowIcon(QtGui.QIcon("utility/logo.png"))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(320, 480))
        self.setMaximumSize(QtCore.QSize(320, 480))
        self.setBaseSize(QtCore.QSize(900, 500))
        font = QtGui.QFont()
        font.setKerning(True)
        self.setFont(font)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.new_experiment = QtWidgets.QFrame(self.centralwidget)
        self.new_experiment.setEnabled(True)
        self.new_experiment.setGeometry(QtCore.QRect(10, 10, 301, 431))
        font = QtGui.QFont()
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.new_experiment.setFont(font)
        self.new_experiment.setAutoFillBackground(True)
        self.new_experiment.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.new_experiment.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.new_experiment.setObjectName("new_experiment")
        self.label_4 = QtWidgets.QLabel(self.new_experiment)
        self.label_4.setGeometry(QtCore.QRect(36, -10, 231, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.label_4.setFont(font)
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_2 = QtWidgets.QLabel(self.new_experiment)
        self.label_2.setGeometry(QtCore.QRect(20, 90, 170, 30))
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.spinBox_all_time = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_all_time.setGeometry(QtCore.QRect(200, 90, 82, 26))
        self.spinBox_all_time.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_all_time.setMinimum(5)
        self.spinBox_all_time.setMaximum(100000000)
        self.spinBox_all_time.setSingleStep(100)
#        self.spinBox_all_time.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.spinBox_all_time.setProperty("value", 240)
        self.spinBox_all_time.setObjectName("spinBox_all_time")
        self.label_5 = QtWidgets.QLabel(self.new_experiment)
        self.label_5.setGeometry(QtCore.QRect(20, 170, 170, 30))
        self.label_5.setTextFormat(QtCore.Qt.AutoText)
        self.label_5.setWordWrap(True)
        self.label_5.setObjectName("label_5")
        self.spinBox_model_max_memory = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_model_max_memory.setGeometry(QtCore.QRect(200, 170, 82, 26))
        self.spinBox_model_max_memory.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_model_max_memory.setMinimum(30)
        self.spinBox_model_max_memory.setMaximum(100000000)
        self.spinBox_model_max_memory.setSingleStep(100)
#        self.spinBox_model_max_memory.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.spinBox_model_max_memory.setProperty("value", 1048576)
        self.spinBox_model_max_memory.setObjectName("spinBox_model_max_memory")
        self.lineEdit_experiment_name = QtWidgets.QLineEdit(self.new_experiment)
        self.lineEdit_experiment_name.setGeometry(QtCore.QRect(20, 50, 261, 21))
        font = QtGui.QFont()
        font.setItalic(False)
        font.setUnderline(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.lineEdit_experiment_name.setFont(font)
        self.lineEdit_experiment_name.setObjectName("lineEdit_experiment_name")
        self.spinBox_max_predict_time = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_max_predict_time.setGeometry(QtCore.QRect(200, 210, 82, 26))
        self.spinBox_max_predict_time.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_max_predict_time.setMinimum(30)
        self.spinBox_max_predict_time.setMaximum(100000000)
        self.spinBox_max_predict_time.setSingleStep(100)
#        self.spinBox_max_predict_time.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.spinBox_max_predict_time.setProperty("value", 40)
        self.spinBox_max_predict_time.setObjectName("spinBox_max_predict_time")
        self.label_6 = QtWidgets.QLabel(self.new_experiment)
        self.label_6.setGeometry(QtCore.QRect(20, 210, 170, 30))
        self.label_6.setTextFormat(QtCore.Qt.AutoText)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.label_12 = QtWidgets.QLabel(self.new_experiment)
        self.label_12.setGeometry(QtCore.QRect(20, 340, 170, 30))
        self.label_12.setTextFormat(QtCore.Qt.AutoText)
        self.label_12.setWordWrap(True)
        self.label_12.setObjectName("label_12")
        self.label_14 = QtWidgets.QLabel(self.new_experiment)
        self.label_14.setGeometry(QtCore.QRect(20, 315, 170, 30))
        self.label_14.setTextFormat(QtCore.Qt.AutoText)
        self.label_14.setWordWrap(True)
        self.label_14.setObjectName("label_14")
        self.btnLoadColumnsDescription = QtWidgets.QPushButton(self.new_experiment)
        self.btnLoadColumnsDescription.setGeometry(QtCore.QRect(200, 345, 82, 26))
        font = QtGui.QFont()
        font.setUnderline(True)
        self.btnLoadColumnsDescription.setFont(font)
        self.btnLoadColumnsDescription.setObjectName("btnLoadColumnsDescription")
        self.btnLoadDataset = QtWidgets.QPushButton(self.new_experiment)
        self.btnLoadDataset.setGeometry(QtCore.QRect(200, 315, 82, 26))
        font = QtGui.QFont()
        font.setUnderline(True)
        self.btnLoadDataset.setFont(font)
        self.btnLoadDataset.setObjectName("btnLoadDataset")
        self.btn_exp_back = QtWidgets.QPushButton(self.new_experiment)
        self.btn_exp_back.setGeometry(QtCore.QRect(20, 390, 71, 23))
        self.btn_exp_back.setObjectName("btn_exp_back")
        self.btnStart = QtWidgets.QPushButton(self.new_experiment)
        self.btnStart.setGeometry(QtCore.QRect(210, 390, 71, 23))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnStart.setFont(font)
        self.btnStart.setObjectName("btnStart")
        self.btn_settings = QtWidgets.QPushButton(self.new_experiment)
        self.btn_settings.setGeometry(QtCore.QRect(100, 390, 100, 23))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.btn_settings.setFont(font)
        self.btn_settings.setObjectName("btn_settings")
        self.label_3 = QtWidgets.QLabel(self.new_experiment)
        self.label_3.setGeometry(QtCore.QRect(20, 130, 170, 30))
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.spinBox_min_accuracy = QtWidgets.QDoubleSpinBox(self.new_experiment)
#        self.spinBox_min_accuracy = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_min_accuracy.setGeometry(QtCore.QRect(200, 130, 82, 26))
        self.spinBox_min_accuracy.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_min_accuracy.setMinimum(0)
        self.spinBox_min_accuracy.setMaximum(100)
        self.spinBox_min_accuracy.setSingleStep(1)
#        self.spinBox_min_accuracy.setStepType(QtWidgets.QAbstractSpinBox.DefaultStepType)
        self.spinBox_min_accuracy.setProperty("value", 65)
#        self.spinBox_min_accuracy.setDisplayIntegerBase(10)
        self.spinBox_min_accuracy.setObjectName("spinBox_min_accuracy")
        
        
        self.label_7 = QtWidgets.QLabel(self.new_experiment)
        self.label_7.setGeometry(QtCore.QRect(20, 250, 170, 30))
        self.label_7.setWordWrap(True)
        self.label_7.setObjectName("label_7")
        self.spinBox_all_time_2 = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_all_time_2.setGeometry(QtCore.QRect(200, 250, 82, 26))
        self.spinBox_all_time_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_all_time_2.setMinimum(30)
        self.spinBox_all_time_2.setMaximum(100000000)
        self.spinBox_all_time_2.setSingleStep(100)
#        self.spinBox_all_time_2.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.spinBox_all_time_2.setProperty("value", 30)
        self.spinBox_all_time_2.setObjectName("spinBox_all_time_2")
        
        #######!!!
        
        self.label_9 = QtWidgets.QLabel(self.new_experiment)
        self.label_9.setGeometry(QtCore.QRect(20, 280, 170, 30))
        self.label_9.setWordWrap(True)
        self.label_9.setObjectName("label_8")
        self.spinBox_iter = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_iter.setGeometry(QtCore.QRect(200, 280, 82, 26))
        self.spinBox_iter.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_iter.setMinimum(1)
        self.spinBox_iter.setMaximum(100000000)
        self.spinBox_iter.setSingleStep(50)
        self.spinBox_iter.setProperty("value", 100)
        self.spinBox_iter.setObjectName("spinBox_iter")
        
        ########!!!
        
        self.search = QtWidgets.QFrame(self.centralwidget)
        self.search.setEnabled(True)
        self.search.setGeometry(QtCore.QRect(10, 10, 301, 431))
        font = QtGui.QFont()
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.search.setFont(font)
        self.search.setAutoFillBackground(True)
        self.search.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.search.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.search.setObjectName("search")
        self.lcd = QtWidgets.QLCDNumber(self.search)
        self.lcd.setGeometry(QtCore.QRect(-14, -10, 311, 131))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.lcd.setFont(font)
        self.lcd.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lcd.setAutoFillBackground(False)
        self.lcd.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lcd.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lcd.setLineWidth(3)
        self.lcd.setMidLineWidth(3)
        self.lcd.setSmallDecimalPoint(False)
        self.lcd.setDigitCount(10)
        self.lcd.setMode(QtWidgets.QLCDNumber.Dec)
        self.lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcd.setProperty("value", 223.0)
        self.lcd.setProperty("intValue", 223)
        self.lcd.setObjectName("lcd")
        self.btn_goto_menu = QtWidgets.QPushButton(self.search)
        self.btn_goto_menu.setEnabled(False)
        self.btn_goto_menu.setGeometry(QtCore.QRect(110, 240, 71, 23))
        self.btn_goto_menu.setObjectName("btn_goto_menu")
        
        self.label = QtWidgets.QLabel(self.search)
        self.label.setGeometry(QtCore.QRect(20, 130, 261, 71))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        
        self.label_8 = QtWidgets.QLabel(self.search)
        self.label_8.setGeometry(QtCore.QRect(20, 160, 261, 71))
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")
        
#        self.btn_search_exit = QtWidgets.QPushButton(self.search)
#        self.btn_search_exit.setEnabled(False)
#        self.btn_search_exit.setGeometry(QtCore.QRect(110, 270, 71, 23))
#        self.btn_search_exit.setObjectName("btn_search_exit")
        self.menu = QtWidgets.QFrame(self.centralwidget)
        self.menu.setEnabled(True)
        self.menu.setGeometry(QtCore.QRect(10, 10, 301, 431))
        font = QtGui.QFont()
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.menu.setFont(font)
        self.menu.setAutoFillBackground(True)
        self.menu.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.menu.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.menu.setObjectName("menu")
        self.btn_menu_search = QtWidgets.QPushButton(self.menu)
        self.btn_menu_search.setGeometry(QtCore.QRect(10, 150, 281, 41))
        self.btn_menu_search.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        font.setKerning(True)
        self.btn_menu_search.setFont(font)
        self.btn_menu_search.setObjectName("btn_menu_search")
        self.btn_menu_exit = QtWidgets.QPushButton(self.menu)
        self.btn_menu_exit.setGeometry(QtCore.QRect(10, 200, 281, 41))
        self.btn_menu_exit.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.btn_menu_exit.setFont(font)
        self.btn_menu_exit.setObjectName("btn_menu_exit")
        self.test = QtWidgets.QFrame(self.centralwidget)
        self.test.setEnabled(True)
        self.test.setGeometry(QtCore.QRect(10, 10, 301, 431))
        font = QtGui.QFont()
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.test.setFont(font)
        self.test.setAutoFillBackground(True)
        self.test.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.test.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.test.setObjectName("test")
        self.label_10 = QtWidgets.QLabel(self.test)
        self.label_10.setGeometry(QtCore.QRect(36, -10, 231, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.label_10.setFont(font)
        self.label_10.setTextFormat(QtCore.Qt.AutoText)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_16 = QtWidgets.QLabel(self.test)
        self.label_16.setGeometry(QtCore.QRect(20, 101, 171, 61))
        self.label_16.setTextFormat(QtCore.Qt.AutoText)
        self.label_16.setWordWrap(True)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.test)
        self.label_17.setGeometry(QtCore.QRect(20, 51, 171, 61))
        self.label_17.setTextFormat(QtCore.Qt.AutoText)
        self.label_17.setWordWrap(True)
        self.label_17.setObjectName("label_17")
        self.btnLoadColumnsDescription_2 = QtWidgets.QPushButton(self.test)
        self.btnLoadColumnsDescription_2.setGeometry(QtCore.QRect(200, 116, 81, 31))
        self.btnLoadColumnsDescription_2.setObjectName("btnLoadColumnsDescription_2")
        self.btnLoadDataset_2 = QtWidgets.QPushButton(self.test)
        self.btnLoadDataset_2.setGeometry(QtCore.QRect(200, 60, 81, 30))
        self.btnLoadDataset_2.setObjectName("btnLoadDataset_2")
        self.pushButton_10 = QtWidgets.QPushButton(self.test)
        self.pushButton_10.setGeometry(QtCore.QRect(60, 390, 71, 23))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButtonNext1_6 = QtWidgets.QPushButton(self.test)
        self.pushButtonNext1_6.setGeometry(QtCore.QRect(140, 390, 121, 23))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonNext1_6.setFont(font)
        self.pushButtonNext1_6.setObjectName("pushButtonNext1_6")
        self.btnSettings_4 = QtWidgets.QPushButton(self.test)
        self.btnSettings_4.setGeometry(QtCore.QRect(10, 180, 281, 41))
        self.btnSettings_4.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.btnSettings_4.setFont(font)
        self.btnSettings_4.setObjectName("btnSettings_4")
        self.test.raise_()
        self.new_experiment.raise_()
        self.search.raise_()
        self.menu.raise_()
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 320, 21))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        # connect
        self.retranslateUi(self)
        self.btn_menu_search.clicked.connect(self.new_experiment.raise_)
        self.btn_menu_exit.clicked.connect(sys.exit)        
        self.btnStart.clicked.connect(self.start_selection)   # начать
        self.btn_exp_back.clicked.connect(self.menu.raise_)
                
        # защита от повторного нажатия после перезахода
        self.btn_goto_menu.clicked.connect( self.btn_goto_menu.setEnabled)
        self.btn_goto_menu.clicked.connect(self.menu.raise_)
        
        
        self.btnLoadDataset.clicked.connect(self.load_dataset_dialog)
        self.btnLoadColumnsDescription.clicked.connect(self.load_column_description_dialog)
        
        # Доп. настройки
#        self.btn_settings.clicked.connect(self.dialog_settings.refresh)
        self.btn_settings.clicked.connect(self.dialog_settings.show)
        
        QtCore.QMetaObject.connectSlotsByName(self)
        
        self.show()

# %% 

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AutoML"))
        self.label_4.setText(_translate("MainWindow", "Новый эксперимент"))
        self.label_2.setText(_translate("MainWindow", "Максимальная длительность поиска (сек)"))
        self.label_5.setText(_translate("MainWindow", "Максимальный объем памяти занимаемый моделью (байт)"))
        self.lineEdit_experiment_name.setText(_translate("MainWindow", self.experiment_name))
        self.label_6.setText(_translate("MainWindow", "Максимальное время выполнения прогноза (мс)"))
        self.label_12.setText(_translate("MainWindow", "Загрузите файл с описанием колонок"))
        self.label_14.setText(_translate("MainWindow", "Загрузите набор данных"))
        self.btnLoadColumnsDescription.setText(_translate("MainWindow", "Загрузить"))
        self.btnLoadDataset.setText(_translate("MainWindow", "Загрузить"))
        self.btn_exp_back.setText(_translate("MainWindow", "Назад"))
        self.btnStart.setText(_translate("MainWindow", "Начать"))
        self.btn_settings.setText(_translate("MainWindow", "Доп. настройки"))
        self.label_3.setText(_translate("MainWindow", "Минимальная точность"))
        self.label_7.setText(_translate("MainWindow", "Максимальное время обучения модели (сек)"))
        self.btn_goto_menu.setText(_translate("MainWindow", "Меню"))
        self.label.setText(_translate("MainWindow", "По завершению поиска результаты будут сохранены в:"))
        self.label_8.setText(_translate("MainWindow", r"C:\Users\maxim\Dropbox\auto_ml\experiment_1"))
        #self.btn_search_exit.setText(_translate("MainWindow", "Выход"))
        self.btn_menu_search.setText(_translate("MainWindow", "Подбор модели"))
        self.btn_menu_exit.setText(_translate("MainWindow", "Выход"))
        self.label_10.setText(_translate("MainWindow", "Тестирование"))
        self.label_16.setText(_translate("MainWindow", "Загрузите набор данных"))
        self.label_17.setText(_translate("MainWindow", "Загрузите модель"))
        self.btnLoadColumnsDescription_2.setText(_translate("MainWindow", "Загрузить"))
        self.btnLoadDataset_2.setText(_translate("MainWindow", "Загрузить"))
        self.pushButton_10.setText(_translate("MainWindow", "Назад"))
        self.pushButtonNext1_6.setText(_translate("MainWindow", "Сделать прогноз"))
        self.btnSettings_4.setText(_translate("MainWindow", "Тестирование"))
        #!!!
        self.label_9.setText(_translate("MainWindow", "Максимальное число итераций оптимизатора"))
      
# %%
     
    def load_dataset_dialog(self):
        fileDlg = QFileDialog(self)
        #fileDlg.setDirectory('./')
        fileDlg.setDirectory(expanduser("~")) 
        #Nikon (*.nef;*.nrw);;Sony (*.arw;*.srf;*.sr2);;All Files (*.*)
        fpath = fileDlg.getOpenFileName(filter="Набор данных (*.csv)")[0] #;;Excel (*.xlsx)
        fpath=os.path.normpath(fpath)
#        print('Dataset path:',fpath)
        
        # if file exist
        if os.path.isfile(fpath):
#            config.handle_dataset_path(fpath)
            self.dataset_path=fpath
            
            _translate = QtCore.QCoreApplication.translate
            self.btnLoadDataset.setText(_translate("MainWindow", "Загружено"))
            font = QtGui.QFont()
            font.setUnderline(False)
            self.btnLoadDataset.setFont(font)            
        
# %%
        
    def load_column_description_dialog(self):
        
        fileDlg = QFileDialog(self)
        from os.path import expanduser
        fileDlg.setDirectory(expanduser("~"))
        fpath = fileDlg.getOpenFileName(filter="Описание столбцов (*.csv)")[0]
        fpath=os.path.normpath(fpath)        
#        print('Columns description file path:',fpath)
        
        #if CD file exist
        if os.path.isfile(fpath):
            self.column_description_path=fpath        
            _translate = QtCore.QCoreApplication.translate
            self.btnLoadColumnsDescription.setText(_translate("MainWindow", "Загружено"))
            font = QtGui.QFont()
            font.setUnderline(False)
            self.btnLoadColumnsDescription.setFont(font)

# %%

    def load_DS_from_disc(self):
        
        from pathlib import Path
        dataset_path = str(Path(self.dataset_path))  
        """
        from numpy import genfromtxt
        self.DS = genfromtxt(dataset_path, delimiter=',')   # ,dtype=None, encoding=None
        """
        import pandas
        self.DS = pandas.read_csv(dataset_path, skiprows=0).dropna(how='any').as_matrix()

    
# %% 
     
    def load_CD_from_disc(self):
        from numpy import genfromtxt
        from pathlib import Path
        column_description_path = str(Path(self.column_description_path))
        numpy_cd = genfromtxt(column_description_path, delimiter=',', 
                              dtype=None, encoding=None)  
        self.CD = numpy_cd.tolist()  

# %%
    def checkbox_state(self, checkbox):
        if checkbox.checkState()==2:
            return True
        elif checkbox.checkState()==0:
            return False    
               
        
    def load_settings_from_gui(self):
        self.lcd_bool = True
        
        self.experiment_name = self.lineEdit_experiment_name.text()
        self.duration = self.spinBox_all_time.value()
        self.min_accuracy = self.spinBox_min_accuracy.value()
        self.max_model_memory = self.spinBox_model_max_memory.value()
        self.max_prediction_time = self.spinBox_max_predict_time.value()
        self.max_train_time = self.spinBox_all_time_2.value()
        self.iterations = self.spinBox_iter.value()
        
        self.used_algorithms = {
        'AdaBoost':self.checkbox_state(self.dialog_settings.checkBox_AdaBoost), 
        'XGBoost':self.checkbox_state(self.dialog_settings.checkBox_XGBoost ), 
        'Bagging(SVС)':self.checkbox_state(self.dialog_settings.checkBox_BaggingSVC ), 
        'MLP':self.checkbox_state(self.dialog_settings.checkBox_MLP ), 
        'HistGB':self.checkbox_state(self.dialog_settings.checkBox_HistGB ), 
        'Ridge':self.checkbox_state(self.dialog_settings.checkBox_Ridge ), 
        'LinearSVC':self.checkbox_state(self.dialog_settings.checkBox_LinearSVC ), 
        'PassiveAggressive':self.checkbox_state(self.dialog_settings.checkBox_PassiveAggressive ), 
        'LogisticRegression':self.checkbox_state(self.dialog_settings.checkBox_LogisticRegression ), 
        'LDA':self.checkbox_state(self.dialog_settings.checkBox_LDA ),  
        'QDA':self.checkbox_state(self.dialog_settings.checkBox_QDA ), 
        'Perceptron':self.checkbox_state(self.dialog_settings.checkBox_Perceptron ),      
        'SVM':self.checkbox_state(self.dialog_settings.checkBox_SVM ), 
        'RandomForest':self.checkbox_state(self.dialog_settings.checkBox_RandomForest ), 
        'ExRandTrees':self.checkbox_state(self.dialog_settings.checkBox_xRandTrees ), 
        'ELM':self.checkbox_state(self.dialog_settings.checkBox_ELM ), 
        'DecisionTree':self.checkbox_state(self.dialog_settings.checkBox_DecisionTree ), 
        'SGD':self.checkbox_state(self.dialog_settings.checkBox_SGD ), 
        'KNeighbors':self.checkbox_state(self.dialog_settings.checkBox_KNeighbors ), 
        'NearestCentroid':self.checkbox_state(self.dialog_settings.checkBox_NearestCentroid ), 
        'GaussianProcess':self.checkbox_state(self.dialog_settings.checkBox_GaussianProcess ), 
        'LabelSpreading':self.checkbox_state(self.dialog_settings.checkBox_LabelSpreading ), 
        'BernoulliNB':self.checkbox_state(self.dialog_settings.checkBox_BernoulliNB ),  
        'GaussianNB':self.checkbox_state(self.dialog_settings.checkBox_GaussianNB ), 
        'DBN':self.checkbox_state(self.dialog_settings.checkBox_DBN ),  
        'FactorizationMachine':self.checkbox_state(self.dialog_settings.checkBox_FactorizationMachine ), 
        'PolynomialNetwork':self.checkbox_state(self.dialog_settings.checkBox_PolynomialNetwork )
        }
                
        self.metric = self.dialog_settings.comboBox_metric.currentText()
        self.validation =self.dialog_settings.comboBox_validation.currentText()
        self.saved_models_count=self.dialog_settings.comboBox_saved_count.currentText()      

# %% 
        
    @pyqtSlot(int)
    def slot_timer(self,value):
        if(self.lcd_bool == True):
            self.lcd.display(value)
        
        
    @pyqtSlot()
    def slot_finish(self):
        self.lcd.display(0)
        self.lcd_bool = False
#        self.btn_goto_menu.setEnabled(True)
        #print(self.sender())#тот кто отправил
        #print("Timer stops")
        
    @pyqtSlot()
    def slot_search_end(self):     
        self.lcd.display(0)
        self.lcd_bool = False
        self.btn_goto_menu.setEnabled(True)
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("MainWindow", "Поиск завершен, результаты сохранены в:"))
        
        
        
         
    def start_timer(self):         
        self.lcd.display(self.duration)
        timer_thread = TimerThread(self)

        #thread.finished.connect(app.exit) #закрыть всё по завершению не уверен
        timer_thread.signal_timer.connect(self.slot_timer)  
        timer_thread.signal_timer_finish.connect(self.slot_finish)
        
        self.signal_start_timer.connect(timer_thread.slot_timer_start) 
        self.signal_start_timer.emit(self.duration)
        
        timer_thread.start()  
        
        
# %%
    
    def check_models(self):       
        for val in self.used_algorithms.values():
            if(val==True):
                return True   
        return False
        
# %%     
        
    def start_selection(self): 
        
        if(self.dataset_path!=None and self.column_description_path!= None) : 
              
            # загрузка данных с форм GUI                  
            self.load_settings_from_gui()
            
            if(self.experiment_name!=''): 
                
                if(self.check_models()==True):
                    self.search.raise_()
                    
                    self.load_DS_from_disc()
                    self.load_CD_from_disc() 
                    ######################################################            
                    self.start_timer()
                    ######################################################           
                    
                    MS_thread = ModelSeletionThread(self)
            
                    #thread.finished.connect(app.exit) #закрыть всё по завершению не уверен
                    #MS_thread.signal_timer.connect(self.slot_timer)  
                    #MS_thread.signal_timer_finish.connect(self.slot_finish)
                    MS_thread.signal_model_selection_finish.connect(self.slot_search_end)
                    
                    self.signal_start_MS.connect(MS_thread.slot_start_model_selection_) 
                    
                    self.signal_start_MS.emit(self.DS, self.CD, self.experiment_name, 
                        self.duration, self.min_accuracy, self.max_model_memory, 
                        self.max_prediction_time, self.max_train_time, 
                        self.used_algorithms, self.metric, self.validation, 
                        self.saved_models_count, self.iterations
                        )
                    
                    MS_thread.start()  
                else:
                    self.warning_models.show()
                
            else:
                self.warning_name.show()
            
        else:
           self.warning_paths.show()

            
            

# %%

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    sys.exit(app.exec_())
