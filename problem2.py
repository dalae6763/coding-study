import cv2 as cv
import numpy as np


bins = np.arange(256).reshape(256,1)


def draw_histogram(img):

    h = np.zeros((img.shape[0], 513), dtype=np.uint8)

    y = np.flipud(h)#problem1.py와 같다

    #draw curve
    hist, bin = np.histogram(img.flatten(), 256, [0, 256])#
    cdf = hist.cumsum()#
    cdf_normalized = cdf * float(hist.max()) / cdf.max()#
    cv.normalize(cdf_normalized, cdf_normalized, 0, 255, cv.NORM_MINMAX)
    hist = np.int32(np.around(cdf_normalized))
    pts = np.int32(np.column_stack((bins, hist)))
    pts += [257, 10]

    cv.line(h, (0+257, 0 + 10), (0+257, 5), (255, 255, 255) )
    cv.line(h, (255+257, 0 + 10), (255+257, 5), (255, 255, 255))
    cv.polylines(h, [pts], False, (255,255,255))

    return y#problem1.py참조
    


img = cv.imread('./data/fruits.jpg', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)




#equ = cv.equalizeHist(gray)
hist, bin = np.histogram(img.flatten(), 256, [0, 256])#np.histogram 함수가 히스토그램을 얻는 함수인데 반환값은 반환값은 256개의 요소를 갖는 배열인 hist와 X축 요소의 값을 나타내는 배열인 bins입니다. 이 함수 이외에도 hist = np.bincount(img.ravel(),minlength=256) 와 같은 더 빠른 함수가 가능합니다.
cdf = hist.cumsum()
cdf_mask = np.ma.masked_equal(cdf,0)# cdf array에서 값이 0인 부분을 mask처리함
cdf_mask = (cdf_mask - cdf_mask.min())*255/(cdf_mask.max()-cdf_mask.min())#History Equalization 공식
cdf = np.ma.filled(cdf_mask,0).astype('uint8')# Mask처리를 했던 부분을 다시 0으로 변환
equ = cdf[gray]


line =  draw_histogram(equ)
result2 = np.hstack((equ, line))
cv.imshow('result2', result2)


cv.waitKey(0)
cv.destroyAllWindows()


