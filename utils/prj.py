import cv2
import numpy as np


def get_vvList(list_data):
    vv_list=list()
    v_list=list()
    for index,i in enumerate(list_data):
        if i>0:
            v_list.append(index)
        else:
            if v_list:
                vv_list.append(v_list)
                v_list=[]
    return vv_list


def detect_txt(img = None, img_path=r'.\img.jpeg', THRESH = 5):
    if img is None:
        img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 改善光照不均
    # 计算所有像素的平均强度
    img = img.astype(float)
    mean_intensity = np.mean(img)

    # 使用闭合重建来估计背景光照水平
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    background = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # 从原始图像中减去背景光照
    img = img - background

    # 添加平均强度
    img = img + mean_intensity
    img = np.clip(img, 0, 255).astype('uint8')
    # 创建一个新窗口并设置窗口的大小
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 600)  # 将窗口大小设置为800x600
    cv2.imshow('image', img)
    cv2.waitKey(0)

    # 局部阈值二值化
    bi_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    # 创建一个新窗口并设置窗口的大小
    cv2.namedWindow('binary image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('binary image', 800, 600)  # 将窗口大小设置为800x600
    cv2.imshow('binary image', bi_img)
    cv2.waitKey(0)

    # 中值滤波
    bi_img = cv2.medianBlur(bi_img, ksize=3)
    # 创建一个新窗口并设置窗口的大小
    cv2.namedWindow('medianBlur', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('medianBlur', 800, 600)  # 将窗口大小设置为800x600
    cv2.imshow('medianBlur', bi_img)
    cv2.waitKey(0)

    # 投影法切行
    base_img = cv2.bitwise_not(bi_img)  # 反转黑白, 使文字为黑, 用于最后切分并输出
    row, col = bi_img.shape
    hor_count = [0]*row  # 保存每行的白色像素数量(即文字像素数量)
    for i in range(row):
        for j in range(col):
            if bi_img[i][j] == 255:
                hor_count[i] = hor_count[i] + 1

    hor_count = np.array(hor_count)
    hor_count[np.where(hor_count < THRESH)] = 0

    hor_count = hor_count.tolist()
    img_white = np.ones(shape=(row, col), dtype=np.uint8) * 255
    for i in range(row):
        pt1 = (col - 1, i)
        pt2 = (col - 1 - hor_count[i], i)
        cv2.line(img_white, pt1, pt2, (0,), 1)
    cv2.imshow('horizontal project', img_white)
    cv2.waitKey(0)

    vv_list = get_vvList(hor_count)
    # for i in vv_list:
    #     img_hor = origin_img[i[0]:i[-1], :, :]
    #     if img_hor.size != 0:
    #         cv2.imshow('文本行', img_hor)
    #         cv2.waitKey(0)
    for i in vv_list:
        img_hor = base_img[i[0]:i[-1], :]
        if img_hor.size != 0:
            cv2.imshow('文本行', img_hor)
            cv2.waitKey(0)


detect_txt()