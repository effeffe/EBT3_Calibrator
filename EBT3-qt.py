#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Filippo Falezza
<filippo dot falezza at outlook dot it>
<fxf802 at student dot bham dot ac dot uk>

Released under GPLv3 and followings
"""

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout

def top_():
    print('top')

def bottom_():
    print('bottom')


def main():
    app = QApplication([])
    app.setStyle('Fusion')

    #label = QLabel('EBT3 Analysis Tool')
    window = QWidget()
    layout = QVBoxLayout()

    top = QPushButton('Top')
    top.clicked.connect(top_)
    layout.addWidget(top)

    bottom = QPushButton('Bottom')
    bottom.clicked.connect(bottom_)
    layout.addWidget(bottom)

    window.setLayout(layout)

    window.show()
    app.exec_()



if __name__ == '__main__':
    main()
