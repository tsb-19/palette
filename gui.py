import sys
import numpy as np
from PyQt5.QtCore import QCoreApplication
from skimage import io, color
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QColor, QImage
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QFileDialog, QApplication, QVBoxLayout, \
    QColorDialog, QProgressDialog

from palette import sample, k_means, lab2rgb, repaint


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.k = 5  # 聚类数
        self.old_lab = np.zeros(self.k)  # 原调色板颜色
        self.pixels = np.zeros(0)  # 原像素
        self.new_pixels = np.zeros(0)  # 重着色后的像素
        self.weights = np.zeros(0)  # 权重

        self.box = QHBoxLayout(self)
        self.color = QVBoxLayout(self)
        self.tool = QWidget(self)
        self.label = QLabel(self)  # 显示图片
        self.open = QPushButton(self)  # 打开按钮
        self.colors = [QPushButton(self) for _ in range(self.k)]  # 调色板
        self.save = QPushButton(self)  # 保存按钮
        self.init()

    def init(self):
        self.setMinimumSize(600, 400)
        self.label.setScaledContents(True)  # 适应图片大小
        self.open.setText("打开")
        self.open.setFixedSize(60, 40)
        self.open.clicked.connect(self.open_clicked)  # 打开点击事件
        self.save.setText("保存")
        self.save.setFixedSize(60, 40)
        self.save.clicked.connect(self.save_clicked)  # 保存点击事件
        self.color.addWidget(self.open)
        self.color.addWidget(self.save)
        # 初始化调色板
        for i in range(self.k):
            self.colors[i].clicked.connect(self.recolor)  # 重着色触发事件
            self.colors[i].setFixedSize(60, 40)
            self.color.addWidget(self.colors[i])

        self.tool.setLayout(self.color)
        self.tool.setMaximumWidth(120)
        self.box.addWidget(self.tool)
        self.box.addWidget(self.label)

        screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.move((screen.width() - self.geometry().width()) // 2, (screen.height() - self.geometry().height()) // 2)
        self.setLayout(self.box)
        self.setWindowTitle('调色板图像重着色')
        self.show()

    def open_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", filter="*.jpg;*.png;*.jpeg;;")  # 获得图片路径
        if len(path) > 0:
            # 进度条
            progress = QProgressDialog(self)
            progress.setWindowTitle("进度")
            progress.setRange(0, 100)
            picture = np.asarray(io.imread(path))[..., :3]  # 将图片读取为数组
            self.pixels = sample(picture, progress)  # 统计像素个数
            self.new_pixels = color.lab2rgb(self.pixels) * 255
            means, self.old_lab, self.weights = k_means(progress, k=self.k)  # 聚类获得调色板颜色
            # 设置调色板颜色
            for i in range(self.k):
                r, g, b = round(means[i][0]), round(means[i][1]), round(means[i][2])
                self.colors[i].setStyleSheet("background-color:" + QColor(r, g, b).name())
            self.label.setPixmap(QPixmap(path))  # 显示图片
            QCoreApplication.processEvents()
            progress.setValue(100)
            progress.close()

    def save_clicked(self):
        img = self.new_pixels.astype("uint8")  # 将数组转换为8bit存储格式
        img = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        path, fmt = QFileDialog.getSaveFileName(self, "保存图片", filter="*.jpg;;*.png;;*.jpeg;;")  # 获得图片存储位置
        if len(path) > 0:
            img.save(path, fmt[2:])

    def recolor(self):
        old = QColor(self.sender().styleSheet()[-7:])
        new = QColorDialog.getColor(initial=old)
        if new.isValid():
            # 进度条
            progress = QProgressDialog(self)
            progress.setWindowTitle("进度")
            progress.setRange(0, 100)
            self.sender().setStyleSheet("background-color:" + new.name())
            old_palette = color.rgb2lab(np.array([old.red(), old.green(), old.blue()]) / 255)  # 调色板原来颜色
            new_palette = color.rgb2lab(np.array([new.red(), new.green(), new.blue()]) / 255)  # 调色板现在颜色
            new_lab = []
            for i in range(self.k):
                c_old = QColor(self.colors[i].styleSheet()[-7:])
                c_lab = color.rgb2lab(np.array([c_old.red(), c_old.green(), c_old.blue()]) / 255)
                # 如果新的调色板亮度不满足递增的要求则进行调整
                if old_palette[0] < c_lab[0] < new_palette[0] or new_palette[0] < c_lab[0] < old_palette[0]:
                    c_lab[0] = new_palette[0]
                    c_new_rgb = lab2rgb(c_lab)
                    c_new = QColor(round(c_new_rgb[0]), round(c_new_rgb[1]), round(c_new_rgb[2]))
                    self.colors[i].setStyleSheet("background-color:" + c_new.name())
                new_lab.append(c_lab)
            # 根据新调色板重着色
            self.new_pixels = repaint(self.pixels, self.weights, self.old_lab, new_lab, progress, k=self.k)
            img = self.new_pixels.astype("uint8")
            img = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap(img))
            QCoreApplication.processEvents()
            progress.setValue(100)
            progress.close()


if __name__ == '__main__':
    application = QApplication(sys.argv)
    widget = Widget()
    sys.exit(application.exec_())
