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
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(285, 560)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(10, 30, 251, 501))
        self.pushButton_3 = QPushButton(self.groupBox_2)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(20, 40, 181, 51))
        self.pushButton_4 = QPushButton(self.groupBox_2)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(20, 110, 181, 51))
        self.pushButton_5 = QPushButton(self.groupBox_2)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(20, 350, 181, 51))
        self.groupBox_3 = QGroupBox(self.groupBox_2)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(10, 180, 211, 161))
        self.comboBox = QComboBox(self.groupBox_3)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(10, 50, 151, 31))
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.pushButton_6 = QPushButton(self.groupBox_3)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setGeometry(QRect(10, 100, 181, 51))
        self.pushButton_7 = QPushButton(self.groupBox_2)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(20, 420, 181, 51))
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
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"2.  Calibration", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"2.1 Corner dection", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"2.2 Find intrinsic", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"2.4 Find the distortion", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"2.3 Find Extrinsic", None))
        self.pushButton_6.setText(QCoreApplication.translate("MainWindow", u"2.3 Find Extrinsic", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"2.5 Show result", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"1", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"2", None))
        self.comboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"3", None))
        self.comboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"4", None))
        self.comboBox.setItemText(4, QCoreApplication.translate("MainWindow", u"5", None))
        self.comboBox.setItemText(5, QCoreApplication.translate("MainWindow", u"6", None))
        self.comboBox.setItemText(6, QCoreApplication.translate("MainWindow", u"7", None))
        self.comboBox.setItemText(7, QCoreApplication.translate("MainWindow", u"8", None))
        self.comboBox.setItemText(8, QCoreApplication.translate("MainWindow", u"9", None))
        self.comboBox.setItemText(9, QCoreApplication.translate("MainWindow", u"10", None))
        self.comboBox.setItemText(10, QCoreApplication.translate("MainWindow", u"11", None))
        self.comboBox.setItemText(11, QCoreApplication.translate("MainWindow", u"12", None))
        self.comboBox.setItemText(12, QCoreApplication.translate("MainWindow", u"13", None))
        self.comboBox.setItemText(13, QCoreApplication.translate("MainWindow", u"14", None))
        self.comboBox.setItemText(14, QCoreApplication.translate("MainWindow", u"15", None))
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
    def onChanged(self, text):
        self.index = int(text)

    def initUI(self):
        self.pushButton_3.clicked.connect(self.Q21)
        self.pushButton_4.clicked.connect(self.Q22)
        self.pushButton_5.clicked.connect(self.Q24)
        self.pushButton_6.clicked.connect(self.Q23)
        self.comboBox.activated[str].connect(self.onChanged)
        self.pushButton_7.clicked.connect(self.Q25)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = My_window()
    ui.show()
    sys.exit(app.exec_())
