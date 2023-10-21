from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QPushButton, QSpacerItem, QSizePolicy,QTextEdit,QFileDialog,QLabel
from PyQt5 import QtWidgets,uic,QtCore
from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QPen
from PyQt5.QtWidgets import QGraphicsView, QGraphicsPixmapItem, QGraphicsScene, QGraphicsItem

from utils import divide
from val import val

class Form(QDialog):
    def __init__(self):
        super(Form, self).__init__()

        self.setWindowTitle("手写中文识别")  # 修改标题
        self.resize(1024, 700)
        self.picture = None
        self.init_ui()


        self.graphicsView.save_signal.connect(self.pushButton_save.setEnabled)
        self.pushButton_cut.clicked.connect(self.pushButton_cut_clicked)
        self.pushButton_save.clicked.connect(self.pushButton_save_clicked)
        self.pushButton_xianshi.clicked.connect(self.boxSelect)
        self.pushButton_shibie.clicked.connect(self.shibie)

        # image_item = GraphicsPolygonItem()
        # image_item.setFlag(QGraphicsItem.ItemIsMovable)
        # self.scene.addItem(image_item)
    def boxSelect(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                                   "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            pixmap.save(r"test.jpg")
            self.picture=r"test.jpg"
            self.graphicsView.scene.clear()
            self.graphicsView.image_item = GraphicsPixmapItem(QPixmap(self.picture))
            self.graphicsView.image_item.setFlag(QGraphicsItem.ItemIsMovable)
            self.graphicsView.scene.addItem(self.graphicsView.image_item)
            size = self.graphicsView.image_item.pixmap().size()
            self.graphicsView.image_item.setPos(-size.width() / 2, -size.height() / 2)
    def shibie(self):
        img_list = divide.divide_text()
        str_list = val(img_list=img_list)
        str_show = '\n'.join(str_list)
        self.text_edit.setPlainText(str_show)

    def init_ui(self):

        background = QLabel(self)
        background.setStyleSheet("background-color: lightblue;")
        background.resize(1224, 700)
        background.move(0, 0)
        background.lower()  # 将背景放在最底层


        self.gridLayout = QGridLayout(self)
        self.pushButton_cut = QPushButton('剪切', self)
        self.pushButton_cut.setCheckable(True)
        self.pushButton_cut.setMaximumSize(QSize(200, 16777215))
        self.gridLayout.addWidget(self.pushButton_cut, 1, 1, 1, 1)
        self.pushButton_cut.setStyleSheet(
            "background-color: rgb(120, 120, 120);"
            "color: white;"
            "border-radius: 5px;"
        )

        self.pushButton_save = QPushButton('保存', self)
        self.pushButton_save.setEnabled(False)
        self.pushButton_save.setMaximumSize(QSize(200, 16777215))
        self.gridLayout.addWidget(self.pushButton_save, 1, 2, 1, 1)
        self.pushButton_save.setStyleSheet(
            "background-color: rgb(120, 120, 120);"
            "color: white;"
            "border-radius: 5px;"
        )

        self.pushButton_xianshi = QPushButton('选取图片', self)
        self.pushButton_xianshi.setCheckable(True)
        self.pushButton_xianshi.setMaximumSize(QSize(200, 16777215))
        self.gridLayout.addWidget(self.pushButton_xianshi, 1, 0, 1, 1)
        self.pushButton_xianshi.setStyleSheet(
            "background-color: rgb(120, 120, 120);"
            "color: white;"
            "border-radius: 5px;"
        )

        self.pushButton_shibie = QPushButton('识别', self)
        self.pushButton_shibie.setCheckable(True)
        self.pushButton_shibie.setMaximumSize(QSize(200, 16777215))
        self.gridLayout.addWidget(self.pushButton_shibie, 1, 3, 1, 1)
        self.pushButton_shibie.setStyleSheet(
            "background-color: rgb(120, 120, 120);"
            "color: white;"
            "border-radius: 5px;"
        )

        self.text_edit = QTextEdit(self)
        self.gridLayout.addWidget(self.text_edit, 0,3, 1, 2)

        self.graphicsView = GraphicsView(self.picture, self)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gridLayout.addWidget(self.graphicsView, 0, 0, 1, 3)


        # 设置按钮的宽度
        button_width = 150
        self.pushButton_cut.setFixedWidth(button_width)
        self.pushButton_save.setFixedWidth(button_width)
        self.pushButton_xianshi.setFixedWidth(button_width)
        self.pushButton_shibie.setFixedWidth(button_width)

        # 设置按钮的高度
        button_height = 40
        self.pushButton_cut.setFixedHeight(button_height)
        self.pushButton_save.setFixedHeight(button_height)
        self.pushButton_xianshi.setFixedHeight(button_height)
        self.pushButton_shibie.setFixedHeight(button_height)

        # 设置文本编辑框的高度
        text_edit_height = 600
        text_edit_width=300
        self.text_edit.setFixedHeight(text_edit_height)
        self.text_edit.setFixedWidth(text_edit_width)
        # 设置 GraphicsView 的大小
        graphics_view_width = 800
        graphics_view_height = 600
        self.graphicsView.setFixedWidth(graphics_view_width)
        self.graphicsView.setFixedHeight(graphics_view_height)

    def pushButton_cut_clicked(self):
        if self.graphicsView.image_item.is_start_cut:
            self.graphicsView.image_item.is_start_cut = False
            self.graphicsView.image_item.setCursor(Qt.ArrowCursor)  # 箭头光标
        else:
            self.graphicsView.image_item.is_start_cut = True
            self.graphicsView.image_item.setCursor(Qt.CrossCursor)  # 十字光标

    def pushButton_save_clicked(self):
        rect = QRect(self.graphicsView.image_item.start_point.toPoint(),
                     self.graphicsView.image_item.end_point.toPoint())
        new_pixmap = self.graphicsView.image_item.pixmap().copy(rect)
        new_pixmap.save(r'test.jpg')
        self.picture = r"test.jpg"
        self.graphicsView.scene.clear()
        self.graphicsView.image_item = GraphicsPixmapItem(QPixmap(self.picture))
        self.graphicsView.image_item.setFlag(QGraphicsItem.ItemIsMovable)
        self.graphicsView.scene.addItem(self.graphicsView.image_item)
        size = self.graphicsView.image_item.pixmap().size()
        self.graphicsView.image_item.setPos(-size.width() / 2, -size.height() / 2)

class GraphicsView(QGraphicsView):
    save_signal = pyqtSignal(bool)

    def __init__(self, picture, parent=None):
        super(GraphicsView, self).__init__(parent)
        self.setBackgroundBrush(QColor(14, 20, 20))

        # 设置放大缩小时跟随鼠标
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.image_item = GraphicsPixmapItem(QPixmap(picture))
        self.image_item.setFlag(QGraphicsItem.ItemIsMovable)
        self.scene.addItem(self.image_item)

        size = self.image_item.pixmap().size()
        # 调整图片在中间
        self.image_item.setPos(-size.width() / 2, -size.height() / 2)

        self.scale(0.4, 0.4)

    def wheelEvent(self, event):
        '''滚轮事件'''
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor

        self.scale(zoomFactor, zoomFactor)

    def mouseReleaseEvent(self, event):
        '''鼠标释放事件'''
        # print(self.image_item.is_finish_cut, self.image_item.is_start_cut)
        if self.image_item.is_finish_cut:
            self.save_signal.emit(True)
        else:
            self.save_signal.emit(False)


class GraphicsPixmapItem(QGraphicsPixmapItem):
    save_signal = pyqtSignal(bool)

    def __init__(self, picture, parent=None):
        super(GraphicsPixmapItem, self).__init__(parent)

        self.setPixmap(picture)
        self.is_start_cut = False
        self.current_point = None
        self.is_finish_cut = False

    def mouseMoveEvent(self, event):
        '''鼠标移动事件'''
        self.current_point = event.pos()
        if not self.is_start_cut or self.is_midbutton:
            self.moveBy(self.current_point.x() - self.start_point.x(),
                        self.current_point.y() - self.start_point.y())
            self.is_finish_cut = False
        self.update()

    def mousePressEvent(self, event):
        '''鼠标按压事件'''
        super(GraphicsPixmapItem, self).mousePressEvent(event)
        self.start_point = event.pos()
        self.current_point = None
        self.is_finish_cut = False
        if event.button() == Qt.MidButton:
            self.is_midbutton = True
            self.update()
        else:
            self.is_midbutton = False
            self.update()

    def paint(self, painter, QStyleOptionGraphicsItem, QWidget):
        super(GraphicsPixmapItem, self).paint(painter, QStyleOptionGraphicsItem, QWidget)
        if self.is_start_cut and not self.is_midbutton:
            # print(self.start_point, self.current_point)
            pen = QPen(Qt.DashLine)
            pen.setColor(QColor(0, 150, 0, 70))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 0, 255, 70))
            if not self.current_point:
                return
            painter.drawRect(QRectF(self.start_point, self.current_point))
            self.end_point = self.current_point
            self.is_finish_cut = True

if __name__ == '__main__':
    import sys

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    app.exec_()
