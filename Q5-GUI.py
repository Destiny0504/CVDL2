# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'designerNJRdNt.ui'
##
## Created by: Qt User Interface Compiler version 5.15.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import cv2
import Q2
import Q5
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(285, 450)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(10, 30, 251, 401))
        self.pushButton_3 = QPushButton(self.groupBox_2)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(20, 40, 181, 51))
        self.pushButton_4 = QPushButton(self.groupBox_2)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(20, 110, 181, 51))
        self.pushButton_5 = QPushButton(self.groupBox_2)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(20, 180, 181, 51))
        self.pushButton_7 = QPushButton(self.groupBox_2)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(20, 320, 181, 51))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 645, 20))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"5.Classification", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"5.1 Show model structure", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"5.2 Show TensorBoard", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"5.3 Test", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"5.4 Data Augmantation", None))
    # retranslateUi
class My_window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(My_window, self).__init__(parent)
        self.setupUi(self)
        self.initUI()
        self.index = 1
        self.img = None
    def Q21(self):
        Q2.Q21()
    def Q22(self):
        Q2.Q22()
    def Q23(self):
        Q2.Q23(self.index)
    def Q24(self):
        Q2.Q24()
    def Q25(self):
        Q2.Q25()

    def initUI(self):
        self.pushButton_3.clicked.connect(self.Q21)
        self.pushButton_4.clicked.connect(self.Q22)
        self.pushButton_5.clicked.connect(self.Q24)
        self.pushButton_7.clicked.connect(self.Q25)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = My_window()
    ui.show()
    sys.exit(app.exec_())
