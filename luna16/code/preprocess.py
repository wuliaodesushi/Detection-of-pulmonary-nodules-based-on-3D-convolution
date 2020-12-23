#coding:utf-8
import numpy as np
import time
from glob import glob
import SimpleITK as itk
from skimage import morphology, measure, segmentation
import h5py
import _pickle as pickle
from config import *
from visual_utils import plot_slices
import pdb

if PROCESS_DONE:
    print('done')
    exit()

def preprocess(num):
    print('start preprocess')
    log_msg("start at {}".format(time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(int(time.time())))))

#     ct_files = glob('{}/subset'+str(num)+'/*.mhd'.format(DATASET_PATH))
    ct_files = glob('../luna/subset'+str(num)+'/*.mhd')
    
    #已经处理过的文件数量
    handled_ids = set([f[-13:-3] for f in glob('{}/*.h5'.format(PREPROCESS_PATH))])
    print('{} total, {} processed'.format(len(ct_files), len(handled_ids)))

    counter = 0
    for f in ct_files:
        seriesuid = f[-14:-4]
#         if seriesuid in handled_ids:
#             print('{} handled'.format(seriesuid))
#             continue

        counter += 1
        print('{} process {}'.format(counter, f))

        itk_img = itk.ReadImage(f)

        img = itk.GetArrayFromImage(itk_img)  # (depth, height, width)
        img = np.transpose(img, (2, 1, 0))  # (width, height, depth)
        '''
        spacing：图像各维度上像素之间的距离（物理层面的，有单位，一般为mm)
        Origin：图像原点的坐标（物理层面的，有单位，一般为mm，与spacing保持一致）
        因为(depth, height, width)，所以要翻转
        '''
        origin = np.array(list(reversed(itk_img.GetOrigin())))
        spacing = np.array(list(reversed(itk_img.GetSpacing())))


        _start_time = time.time()
        '''
        pixels 不为0的像素点数量
        cover_ratio  不为0的像素点数量/像素点总数
        '''
        img, pixels = get_lung_img(img)  
        duration = time.time() - _start_time
        cover_ratio = pixels / np.prod(img.shape)

        meta = {
            'seriesuid': seriesuid,
            'shape': img.shape,
            'origin': origin,
            'spacing': spacing,
            'pixels': pixels,
            'cover_ratio': cover_ratio,
            'process_duration': duration,
        }
        pdb.set_trace()
        save_to_numpy(seriesuid, img, meta)

        log_msg(meta)

    print('all preprocess done')

def log_msg(msg):
    
    with open(MSG_LOG_FILE, 'a') as f:
        f.write(str(msg) + '\n')
    print(msg)

def save_to_numpy(seriesuid, img, meta):
    file = '{}/{}'.format(PREPROCESS_PATH, seriesuid)

    with h5py.File(file + '.h5', 'w') as hf:
        hf.create_dataset('img', data=img)

    with open(file + '.meta', 'wb') as f:
        pickle.dump(meta, f)

def get_lung_img(img):
    origin_img = img.copy()
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'origin')

    # binary
    img = img < BINARY_THRESHOLD
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'binary')
        

    # clear_border
    #清除连接到标签图像边界的对象。
    '''
    clear_border(labels, buffer_size=0, bgval=0, in_place=False)主要作用是清除二值图像中边界的1值。
    1值表示白，0值表示黑
    '''
    for c in range(img.shape[2]):
        img[:, :, c] = segmentation.clear_border(img[:, :, c])
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'clear_border')

    
    # keep 2 lagest connected graph
    
    #标记整数数组的连接区域。两个像素是相邻的，并且具有相同的值。
    '''
    skimage.measure.label(input, neighbors = None, background = None, return_num = False, connectivity = None)
    Parameters:
    input : Image to label 需要被标记的图片，输入的数据结构不作要求
    neighbors : 这个参数将被移除，被下面的connectivity替代。可以忽略不看
    background : 选择背景像素，指定像素作为背景，全部相同像素标记为0
    return_num : 是一个bool值，如果为True的话返回值是一个元组（labels ，num ）；如果为False的话就只返回labels
    connectivity : Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. 
    Accepted values are ranging from 1 to input.ndim. If None, a full connectivity of input.ndim is used. [int, optional]。
    如果input是一个二维的图片，那么connectivity的值范围选择{1,2}，如果是None则默认是取最高的值，
    对于二维来说，当connectivity=1时代表4连通，当connectivity=2时代表8连通.
    Returns:
    labels : 和input形状一样，但是数值是标记号，所以这是一个已经标记的图片
    num : 标记的种类数，如果输出0则只有背景，如果输出2则有两个种类或者说是连通域
    '''
    labels = measure.label(img)
    
    #测量标记的图像区域的属性。
    '''
    如果想分别上面的的每一个连通区域进行操作，比如计算面积、外接矩形、凸包面积等，则需要调用measure子模块的regionprops（）函数。
    返回所有连通区块的属性列表，常用的属性列表如下表：
    属性名称    类型    描述
    area    int    区域内像素点总数
    bbox    tuple    边界外接框(min_row, min_col, max_row, max_col)
    centroid    array　　    质心坐标
    convex_area    int    凸包内像素点总数
    convex_image    ndarray    和边界外接框同大小的凸包　　
    coords    ndarray    区域内像素点坐标
    Eccentricity     float    离心率
    equivalent_diameter     float    和区域面积相同的圆的直径
    euler_number    int　　    区域欧拉数
    extent     float    区域面积和边界外接框面积的比率
    filled_area    int    区域和外接框之间填充的像素点总数
    perimeter     float    区域周长
    label    int    区域标记
    '''
    regions = measure.regionprops(labels)

    labels = [(r.area, r.label) for r in regions]    
    if len(labels) > 2:
        labels.sort(reverse=True)
        '''
        按照连通图的面积排序（从大到小）
        背景，一个肺，另一个肺，其他
        '''
        max_area = labels[2][0]
        for r in regions:
            if r.area < max_area:
                '''
                将小于肺的连通图置为黑色
                '''
                for c_item in r.coords:
                    img[c_item[0], c_item[1], c_item[2]] = 0
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'keep 2 lagest connected graph')

    # erosion
    '''
    函数：skimage.morphology.erosion(image, selem=None）
    selem表示结构元素，用于设定局部区域的形状和大小。
    和膨胀相反的操作，将0值扩充到邻近像素。扩大黑色部分，减小白色部分。可用来提取骨干信息，去掉毛刺，去掉孤立的像素。
    注意，如果处理图像为二值图像（只有0和1两个值），则可以调用：
    skimage.morphology.binary_erosion(image, selem=None）
    '''
    # img = morphology.erosion(img, selem=np.ones((2, 2, 2)))
    # if DEBUG_PREPROCESS_PLOT:
    #    plot_slices(img, 'erosion')

    # closing
    #返回图像的灰度形态闭合。
    '''
    函数：skimage.morphology.closing(image, selem=None）    
    selem表示结构元素，用于设定局部区域的形状和大小。    
    先膨胀再腐蚀，可用来填充孔洞。
    注意，如果处理图像为二值图像（只有0和1两个值），则可以调用：
    skimage.morphology.binary_closing(image, selem=None）
    用此函数比处理灰度图像要快
    '''
    img = morphology.closing(img, selem=np.ones((4, 4, 4)))
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'closing')

    # dilation
    #返回图像的灰度形态膨胀。
    '''
    膨胀（dilation)    
    原理：一般对二值图像进行操作。找到像素值为1的点，将它的邻近像素点都设置成这个值。.
    1值表示白，0值表示黑，因此膨胀操作可以扩大白色值范围，压缩黑色值范围。一般用来扩充边缘或填充小的孔洞。    
    功能函数：skimage.morphology.dilation(image, selem=None）    
    selem表示结构元素，用于设定局部区域的形状和大小。
    注意，如果处理图像为二值图像（只有0和1两个值），则可以调用：
    skimage.morphology.binary_dilation(image, selem=None）
    用此函数比处理灰度图像要快。
    '''
    img = morphology.dilation(img, selem=np.ones((16, 16, 16)))
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'dilation')

    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img * origin_img, 'final')

    return img * origin_img, np.sum(img != 0)

if __name__ == '__main__':
    for i in range(0,10):
        preprocess(i)
