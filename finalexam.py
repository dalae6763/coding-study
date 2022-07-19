import copy
import cv2
import numpy as np

# 가우시안 필터 제거하기 + 평활화 제거하기 + 히스토그램 비교 + 파일 저장!
# 물체의 흐름을 그리기 위한 코드
def drawFlow(img, flow, thresh=2, stride=8):
    h, w = img.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow2 = np.int32(flow)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            dx, dy = flow2[y, x]
            if mag[y, x] > thresh:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                cv2.line(img, (x, y), (x + dx, y + dy), (255, 0, 0), 1)

# 2
src = cv2.VideoCapture('./data/기말_동영상.mp4')  # dataset 불러오기
height, width = (int(src.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(src.get(cv2.CAP_PROP_FRAME_WIDTH)))

ret, frame = src.read()
imgP = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # flow에 활용할 grayscale의 이미지

# Flow에 활용할 option들을 while문 밖에서 선언
TH = 2
AREA_TH = 50
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
temp = None  # 이전 프레임 복사를 위한 변수
count = 0
# 4
while True:
    retval, frame = src.read()  # dataset(동영상) 읽어서 frame에 반환
    imgC = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # step1
    flow = cv2.calcOpticalFlowFarneback(imgP, imgC, None, **params)
    drawFlow(frame, flow, TH)  # 메소드에 매게변수를 던져서 flow를 그린다
    # step2
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    ret, blmage = cv2.threshold(mag, TH, 255, cv2.THRESH_BINARY)
    blmage = blmage.astype(np.uint8)  # 타입을 바꾸어 준다
    contours, hierarchy = cv2.findContours(blmage, mode, method)  # image의 특징점을 찾아 반환
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > AREA_TH:
            x, y, width, height = cv2.boundingRect(cnt)  # roi 지정
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)  # frame에 roi를 그려주기 빨강색

    # histogram을 위한 선언
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, cv2.CV_32F)
    # dst = cv2.equalizeHist(gray)  # 평활화 제거!!

    # histogram 계산
    H1 = cv2.calcHist(images=[gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])  # 현재 히스토그램
    H1_b = cv2.calcHist(images=[temp], channels=[0], mask=None, histSize=[256], ranges=[0, 256])  # 이전 프래임
    diff = H1 - H1_b #히스토그램 차이
    print(np.abs(diff).sum())

    #비교 후 text출력
    if np.abs(diff).sum() >= 110000:
        text = 'Shot Changed'
        org = (30, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, org, font, 1, (255, 200, 0), 3)
        cv2.imwrite('./image{}.png'.format(count),imgC)
        count += 1
    else:
        text = 'Not Changed'
        org = (30, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, org, font, 1, (0, 0, 255), 3)

    temp = gray.copy()  # temp라는 변수에 dst(평활화의 결과를 복사 후 활용) 이전 프레임의 히스토그램을 그리는데 활용
    cv2.imshow('frame', frame)
    imgP = imgC.copy()

    key = cv2.waitKey(25)
    if key == 27:
        break

if src.isOpened():
    src.release()
cv2.destroyAllWindows()