from PyQt5.QtWidgets import *

class MainWidget(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        btn = QPushButton("GG")
        layout.addWidget(btn)

        btn.clicked.connect(lambda b: btn.setText("Why Click me"))


class MainWin(QMainWindow):

    def __init__(self):
        super(QMainWindow,self).__init__(None)
        main = MainWidget()
        self.setCentralWidget(main)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = MainWin()
    win.resize(700, 800)
    win.show()
    sys.exit(app.exec_())
