from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PyQt6.QtCore import Qt


def create_icon_pixmap():
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor(220, 220, 220))
    painter.setFont(QFont("Sans", 36, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "A")
    painter.end()
    return pixmap


def create_tray(parent, on_toggle, on_quit):
    icon = QIcon(create_icon_pixmap())
    tray = QSystemTrayIcon(icon, parent)
    tray.setToolTip("AndrewCLI")

    menu = QMenu()
    menu.addAction("Ask Andrew").triggered.connect(on_toggle)
    menu.addSeparator()
    menu.addAction("Quit").triggered.connect(on_quit)
    tray.setContextMenu(menu)

    tray.activated.connect(lambda _: on_toggle())
    return tray
