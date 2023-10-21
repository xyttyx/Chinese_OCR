import cv2
import numpy as np
import statistics  # 中位数


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
    # 最后一行
    if len(v_list) > 0:
        vv_list.append(v_list)
    return vv_list


'''
img:投影法将依据它产生切割行的切割线
base_img:根据img获得切割线后, 将在base_img上切割
THRESH:小于THRESH割像素的行被视为空白分割行
返回值:两个列表,前者每个元素是列表,保存了一个切割后的文本行图像;后者的
    每个元素保存了前者对应的切割后的文本行的行高
'''
def slice(img, base_img=None, THRESH=15):
    if base_img is None:
        base_img = img
    # 局部阈值二值化
    bi_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # 中值滤波
    bi_img = cv2.medianBlur(bi_img, ksize=3)

    # 投影法切行
    row, col = bi_img.shape
    hor_count = [0] * row  # 保存每行的白色像素数量(即文字像素数量)
    for i in range(row):
        for j in range(col):
            if bi_img[i][j] == 255:
                hor_count[i] = hor_count[i] + 1

    hor_count = np.array(hor_count)
    hor_count[np.where(hor_count < THRESH)] = 0

    hor_count = hor_count.tolist()

    vv_list = get_vvList(hor_count)

    img_list = []  # 输出的分行的图像列表
    rows_height = []  # 每行的行高

    for i in vv_list:
        img_hor = base_img[i[0]:i[-1], :]

        if img_hor.size != 0:
            img_list.append([img_hor])
            rows_height.append(i[-1] - i[0] + 1)

    return img_list, rows_height



'''
input:img-输入待划分行的图像,或使用img_path指明图像路径,THRESH-小于THRESH
    个像素的行将被视为空白行
output:一个列表,每个元素是一个划分后的文本行图像
'''
def divide_text(img = None, img_path=r'.\test.jpg', THRESH = 15):
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

    # 对比度增强
    # 定义alpha（对比度控制）和beta（亮度控制）
    alpha = 1.5
    beta = 0

    # 使用cv2.convertScaleAbs()调整对比度和亮度
    base_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 切割文本行
    img_list, rows_height = slice(img, base_img, THRESH=THRESH)

    '''
    由于投影法分出的文本行图像可能过度分割出一些无效的行,
    或者分不出来,形成一个包含多行文本的图像,故取行高的中位数,
    flag标记分出的所有文本行图像:0-正常,1-过小,2-过大,
    定义大于中位数1.8倍为过大,小于中位数0.2倍为过小.
    删除过小的,二次划分过大的.
    '''
    median_height = statistics.median(rows_height)
    flag = [0]*len(img_list)
    for i, height in enumerate(rows_height):
        if height <= median_height*0.2:
            flag[i] = 1
        elif height >= median_height * 1.8:
            flag[i] = 2

    large_idx = [i for i, item in enumerate(flag) if item == 2]
    for idx in large_idx:
        large_lines = img_list[idx][0]
        # 腐蚀膨胀
        # 创建核结构
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构
        kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 十字形结构
        # 腐蚀
        img_erode = cv2.erode(large_lines, kernel3, iterations=2)

        # 膨胀
        img_dilate = cv2.dilate(img_erode, kernel1, iterations=3)

        # 二次切割文本行
        imgs_list, heights = slice(img=img_dilate, base_img=large_lines, THRESH=3*THRESH)

        del_idx = []
        for i, height in enumerate(heights):
            # 仅删除过小的文本行划分
            if height <= median_height * 0.2:
                del_idx.append(i)

        for i in reversed(del_idx):
            del imgs_list[i]

        # 用新划分的文本行图像列表替换之前的文本行图像
        img_list[idx] = [item[0] for item in imgs_list]

    # 将最终划分结果整合到一个列表,每个元素是一个划分后的文本行图像
    result_imgs = []
    for i, imgs in enumerate(img_list):
        if flag[i] == 1:
            continue  # 过小图像跳过
        for img in imgs:
            result_imgs.append(img)

    return result_imgs


# imgs = detect_txt(img_path='./zh.jpg',THRESH=10)
# for img in imgs:
#     cv2.imshow('show', img)
#     cv2.waitKey(0)