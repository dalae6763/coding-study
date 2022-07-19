import cv2
import numpy as np
import matplotlib.pylab as plt

bins = np.arange(256).reshape(256,1)


def draw_histogram(img):

    h = np.zeros((img.shape[0], 513), dtype=np.uint8)

    hist_item = cv2.calcHist([img],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0+10),(x,y+10),(255,255,255))

    cv2.line(h, (0, 0 + 10), (0, 5), (255, 255, 255) )
    cv2.line(h, (255, 0 + 10), (255, 5), (255, 255, 255))
    y = np.flipud(h)

    #draw curve
    hist, bin = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cv2.normalize(cdf_normalized, cdf_normalized, 0, 255, cv2.NORM_MINMAX)
    hist = np.int32(np.around(cdf_normalized))
    pts = np.int32(np.column_stack((bins, hist)))
    pts += [257, 10]

    cv2.line(h, (0+257, 0 + 10), (0+257, 5), (255, 255, 255) )
    cv2.line(h, (255+257, 0 + 10), (255+257, 5), (255, 255, 255))
    cv2.polylines(h, [pts], False, (255,255,255))

    return y


#img = cv2.imread('./data/lena.jpg', cv2.IMREAD_COLOR)
#gray = cv2.imread(img,cv2.COLOR_BGR2GRAY)


gray = cv2.imread('./data/lena.jpg',cv2.IMREAD_GRAYSCALE).copy()
line =  draw_histogram(gray)
result1 = np.hstack((gray, line))
cv2.imshow('result1', result1)  


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
equ = clahe.apply(gray)


line =  draw_histogram(equ)
result2 = np.hstack((equ, line))
cv2.imshow('result2', result2)


cv2.waitKey(0)
cv2.destroyAllWindows()