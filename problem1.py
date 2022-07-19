import cv2
import numpy as np
#왜 numpy인가 배열의 크기가 작으면 리스트나 튜플을 사용하면되지만 
#numpy 배열은 데이터의 크기가 커질수록 저장 및 가공을 하는데 효율성을 보장한다 


bins = np.arange(256).reshape(256,1)#배열 256까지 생성후 1행으로 0부터 255까지 재배열


def draw_histogram(img):


    h = np.zeros((img.shape[0], 513), dtype=np.uint8)#이미지를 가운데로 겹치도록 설정?
    hist, bin = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()#numpy 배열을 1차원 배열로 변경한 후, 각 멤버값을 누적하여 더한 값을 멤버로하는 numpy 1차원배열을 생성합니다.
    cdf_normalized = cdf * float(hist.max()) / cdf.max()#정규화 과정인데 수식은 정확히 모르겠다?
    cv2.normalize(cdf_normalized, cdf_normalized, 0, 255, cv2.NORM_MINMAX)#?
    hist = np.int32(np.around(cdf_normalized))#int32로 변환 후 cdf_normalized 소수점자리를 반올림
    pts = np.int32(np.column_stack((bins, hist)))#1 차원 배열을 2 차원 배열에 열로 쌓습니다.
    pts += [257, 10]#?정확히 어느 배열에 추가하는건지?
    y = np.flipud(h)#상하반전

    cv2.line(h, (0+257, 0 + 10), (0+257, 5), (255, 255, 255) )#라인그리기
    cv2.line(h, (255+257, 0 + 10), (255+257, 5), (255, 255, 255))#라인그리기
    cv2.polylines(h, [pts], False, (255,255,255))#다각형그리기

    return y


img = cv2.imread('./data/lena.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


line =  draw_histogram(gray)
#result1 = np.hstack((gray, line))#이미지를 나란히 쌓기 
cv2.imshow('img', gray)
cv2.imshow('result1', line)



cv2.waitKey(0)
cv2.destroyAllWindows()