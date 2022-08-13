import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from inference import summarization_pipline
from class_clust_infer import report_pipline
from  Ui_MainWindow import Ui_MainWindow
class MainWindow():
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui =Ui_MainWindow()

        # self.ui =
        self.ui.setupUi(self.main_win)
        self.ui.textEdit.setText('')
        self.ui.pushButton.clicked.connect(self.Generate_summary)

    def Generate_summary(self):
        plaint = self.ui.textEdit.toPlainText()
        # Inference
        summary = summarization_pipline(plaint)
        self.ui.textEdit_2.setText(summary)

        Similarity_score, Summary_category, Original_category, Summary_cluster, Original_cluster = report_pipline(plaint, summary)
        print(Similarity_score, Summary_category, Original_category, Summary_cluster, Original_cluster)
        # Evaluation Results
        self.ui.label_3.setText('Similarity score: '+str(Similarity_score))
        self.ui.label_8.setText('Original cluster: '+str(Original_cluster))
        self.ui.label_9.setText('Summary cluster: '+str(Summary_cluster))
        self.ui.label_5.setText('Original category: '+str(Original_category))
        self.ui.label_6.setText('Summary category: '+str(Summary_category))


    def show(self):
        self.main_win.show()

if __name__ =="__main__":
    app = QApplication(sys.argv)
    main_win =MainWindow()
    main_win.show()
    sys.exit(app.exec())
