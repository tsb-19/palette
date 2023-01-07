from PyQt5.QtCore import QCoreApplication
from scipy.interpolate import RegularGridInterpolator
from skimage import color
import numpy as np

b = 16  # bin的参数
g = 12  # grid的参数
bins_color = np.zeros((b, b, b, 3))  # bin的颜色均值
bins_count = np.zeros((b, b, b))  # bin内的像素数量


# 手动实现lab转rgb，trunc=True返回截断结果，trunc=False返回直接计算结果，用于求解色域边界
def lab2rgb(lab, trunc=True):
    def t(n):
        return n ** 3 if n > 6 / 29 else 3 * ((6 / 29) ** 2) * (n - 4 / 29)

    def gamma(n):
        return 1.055 * (n ** (1 / 2.4)) - 0.055 if n > 0.0031308 else n * 12.92

    X = 0.95047 * t((lab[0] + 16) / 116 + lab[1] / 500)
    Y = 1.0 * t((lab[0] + 16) / 116)
    Z = 1.08883 * t((lab[0] + 16) / 116 - lab[2] / 200)
    R = gamma(3.2406 * X + -1.5372 * Y + -0.4986 * Z) * 255
    G = gamma(-0.9689 * X + 1.8758 * Y + 0.0415 * Z) * 255
    B = gamma(0.0557 * X + -0.2040 * Y + 1.0570 * Z) * 255
    rgb = [R, G, B]
    if trunc:
        for i in range(3):
            if rgb[i] < 0:
                rgb[i] = 0
            elif rgb[i] > 255:
                rgb[i] = 255
    return rgb


# 第一种情况的的色域边界求解，交点在线段延长线上
def boundary1(x, x0):
    l = 0
    r = 512
    d = (x0 - x) / np.linalg.norm(x0 - x)
    for i in range(10):
        m = (l + r) / 2
        test = lab2rgb(m * d + x0, trunc=False)
        illegal = False in [0 <= _ <= 255 for _ in test]
        if illegal:
            r = m
        else:
            l = m
    return l * d + x0


# 第二种情况的的色域边界求解，交点在线段上
def boundary2(new_c, x0):
    l = 0
    r = 1
    d = (x0 - new_c) / np.linalg.norm(x0 - new_c)
    for i in range(10):
        m = (l + r) / 2
        test = lab2rgb(m * d + new_c, trunc=False)
        illegal = False in [0 <= _ <= 255 for _ in test]
        if illegal:
            r = m
        else:
            l = m
    return l * d + new_c


# 单个颜色的f转换函数
def f(x, old_c, new_c):
    x0 = x + new_c - old_c
    illegal = False in [0 <= _ <= 255 for _ in lab2rgb(x0, trunc=False)]
    xb = np.array(boundary2(new_c, x0) if illegal else boundary1(x, x0))
    cb = np.array(boundary1(old_c, new_c))
    if np.linalg.norm(xb - x) > np.linalg.norm(cb - old_c):
        new_x = x + np.linalg.norm(new_c - old_c) * ((xb - x) / np.linalg.norm(xb - x))
    else:
        new_x = x + (xb - x) * np.linalg.norm(new_c - old_c) / np.linalg.norm(cb - old_c)
    return new_x


# 快速rbf函数
def rbf(pixels, weights, old_lab, new_lab, progress, k=5):
    progress.setLabelText("计算网格中……")
    progress.show()
    # 先计算grid的新颜色
    grid = np.zeros((g + 1, g + 1, g + 1, 3))
    for x in range(g + 1):
        for y in range(g + 1):
            for z in range(g + 1):
                rgb = np.array([255 / g * x, 255 / g * y, 255 / g * z])
                lab = color.rgb2lab(rgb / 255)
                for kk in range(k):
                    grid[x][y][z] += weights[x][y][z][kk] * f(lab, old_lab[kk], new_lab[kk])  # 权重事先计算，调用单颜色计算函数
        QCoreApplication.processEvents()
        progress.setValue(x / g * 100)
    # 每个像素的颜色采用grid三线性插值
    x = np.linspace(0, g, g + 1)
    y = np.linspace(0, g, g + 1)
    z = np.linspace(0, g, g + 1)
    interp = RegularGridInterpolator((x, y, z), grid)
    return interp(color.lab2rgb(pixels) * g)  # 返回所有新像素颜色


# 采样函数
def sample(image, progress):
    progress.setLabelText("采样中……")
    progress.show()
    lab = color.rgb2lab(image / 255)
    # 将每个像素归入对应的bin中
    for i in range(image.shape[0]):
        QCoreApplication.processEvents()
        progress.setValue(i / image.shape[0] * 100)
        for j in range(image.shape[1]):
            pos = image[i][j] / 255 * b
            x, y, z = min(int(pos[0]), b-1), min(int(pos[1]), b-1), min(int(pos[2]), b-1)
            bins_count[x][y][z] += 1
            bins_color[x][y][z] += lab[i][j]
    # 计算每个bin的颜色均值
    for x in range(b):
        for y in range(b):
            for z in range(b):
                if bins_count[x][y][z] > 0:
                    bins_color[x][y][z] /= bins_count[x][y][z]
    QCoreApplication.processEvents()
    progress.setValue(100)
    progress.close()
    return lab


# 聚类函数
def k_means(progress, k=5, sigma=80):
    progress.setLabelText("聚类中……")
    progress.show()
    means = np.zeros((k + 1, 3))  # 增加一个黑色聚类
    w = np.array(bins_count)
    for n in range(k):
        pos = np.unravel_index(np.argmax(w), w.shape)
        means[n] = bins_color[pos[0]][pos[1]][pos[2]]  # 每次取权值最大的点作为中心点
        for x in range(b):
            for y in range(b):
                for z in range(b):
                    # 更新权重
                    norm = np.linalg.norm(means[n] - bins_color[x][y][z])
                    w[x][y][z] *= (1 - np.exp(-np.power(norm, 2) / np.power(sigma, 2)))
    # 迭代轮数固定为10
    for i in range(10):
        QCoreApplication.processEvents()
        progress.setValue((i + 1) * 10)
        groups_color = np.zeros((k + 1, 3))
        groups_count = np.zeros(k + 1)
        # 每个bin内的所有像素按照颜色均值归于相应类
        for x in range(b):
            for y in range(b):
                for z in range(b):
                    distances = [np.linalg.norm(bins_color[x][y][z] - means[j]) for j in range(k + 1)]
                    index = np.argmin(distances)
                    groups_color[index] += bins_color[x][y][z] * bins_count[x][y][z]
                    groups_count[index] += bins_count[x][y][z]
        # 更新每类中心点颜色（黑色除外）
        for j in range(k):
            if groups_count[j] > 0:
                means[j] = groups_color[j] / groups_count[j]
            else:
                means[j] = np.zeros(3)
    means = np.flipud(means[np.lexsort((means[:, 2], means[:, 1], means[:, 0]))])  # 按照亮度降序排列
    old_lab = [_ for _ in means[:-1]]  # 初始调色板颜色（去掉黑色）
    phi = np.zeros((k, k))
    # 计算初始调色板中所有点对的均值
    sigma = 0
    for i in range(k):
        for j in range(k):
            sigma += np.linalg.norm(old_lab[i] - old_lab[j])
    sigma /= (k * k)
    # 矩阵求逆计算λ
    for i in range(k):
        for j in range(k):
            phi[i][j] = np.exp(-np.power(np.linalg.norm(old_lab[i] - old_lab[j]), 2) / (2 * sigma * sigma))
    lbd = np.linalg.inv(phi)
    # 计算grid的颜色权重，负数clamp到0，再进行归一化
    weights = np.zeros((g + 1, g + 1, g + 1, k))
    for x in range(g + 1):
        for y in range(g + 1):
            for z in range(g + 1):
                rgb = np.array([255 / g * x, 255 / g * y, 255 / g * z])
                lab = color.rgb2lab(rgb / 255)
                for i in range(k):
                    for j in range(k):
                        weights[x][y][z][i] += lbd[i][j] * np.exp(-np.power(np.linalg.norm(lab - old_lab[j]), 2) / (2 * sigma * sigma))
                    weights[x][y][z][i] = max(0, weights[x][y][z][i])
                if np.sum(weights[x][y][z]) != 0:
                    weights[x][y][z] /= np.sum(weights[x][y][z])
    means = np.array([lab2rgb(_) for _ in means])
    return means[:-1], old_lab, weights


# 图像重着色
def repaint(pixels, weights, old_lab, new_lab, progress, k=5):
    progress.setLabelText("亮度重着色中……")
    progress.show()
    new_pixels = np.array(pixels)
    # 增加亮度边界
    old_l = [100] + [_[0] for _ in old_lab] + [0]
    new_l = [100] + [_[0] for _ in new_lab] + [0]
    # L空间调节
    for i in range(new_pixels.shape[0]):
        for j in range(new_pixels.shape[1]):
            lc = new_pixels[i][j][0]
            for kk in range(k + 1):
                if old_l[kk] >= lc >= old_l[kk + 1]:
                    # 新亮度由线性插值得到，特殊情况取均值
                    if old_l[kk + 1] == old_l[kk]:
                        new_pixels[i][j][0] = 0.5 * (new_l[kk] + new_l[kk + 1])
                    else:
                        t = (old_l[kk] - lc) / (old_l[kk] - old_l[kk + 1])
                        new_pixels[i][j][0] = t * new_l[kk + 1] + (1 - t) * new_l[kk]
                    break
        QCoreApplication.processEvents()
        progress.setValue(i / new_pixels.shape[0] * 100)
    QCoreApplication.processEvents()
    progress.setValue(100)
    progress.close()
    # 调用rbf函数ab空间调节
    new_pixels[:, :, -2:] = rbf(pixels, weights, old_lab, new_lab, progress, k=k)[:, :, -2:]
    return color.lab2rgb(new_pixels) * 255
