import cv2
import numpy as np

cap = cv2.VideoCapture(0) #ë‚´ì–¼êµ´
#cap = cv2.VideoCapture(0) #ì¹´ë©”ë¼
temp = None #ë³µì‚¬í•  tempë³€ìˆ˜ë¥¼ whileë¬¸ ë°–ì—ì„œ ì´ˆê¸°í™”...
while True:
    retval, frame = cap.read()  # camera imageë¥¼ ì½ì–´ì„œ frameì— ë°˜í™˜
    #ì»¬ëŸ¬ë²”ìœ„ì— ë”°ë¥¸ ì˜ì—­ë¶„í• 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # ì»¬ëŸ¬ë²”ìœ„ì— ë”°ë¥¸ ì˜ì—­ë¶„í• ì„ ìœ„í•´ hsvë¡œ ë³€í™˜
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # ëª¨í´ë¡œì§€ì—°ì‚°ì„ ìœ„í•´ grayë¡œ ë³€í™˜
    lowerb = (0,40,0)
    upperb = (20,180,255)
    dst = cv2.inRange(hsv,lowerb,upperb) #ì»¬ëŸ¬ ë²”ìœ„ì— ë”°ë¥¸ ë¶„í• ì˜ìƒ
    #ì¹¨ì‹, íŒ½ì°½ì„ ìœ„í•œ ëª¨í´ë¡œì§€ ì—°ì‚°
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))  # ëª¨í´ë¡œì§€ ì—°ì‚°ì˜ ê²°ê³¼ë¥¼ ì›ìœ¼ë¡œ ë°˜í™˜
    opening = cv2.morphologyEx(dst,cv2.MORPH_OPEN,kernel,iterations=10) # hsvë¡œ ë¶ˆëŸ¬ì™€ì„œ OPENIGì—°ì‚° 10ë²ˆ ì§„í–‰
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE, kernel,iterations=10)  # hsvë¡œ ë¶ˆëŸ¬ì™€ì„œ OPENIGì—°ì‚° 10ë²ˆ ì§„í–‰
    gradient = cv2.morphologyEx(closing,cv2.MORPH_GRADIENT,kernel,iterations=5) # ëª¨í´ë¡œì§€ ì—°ì‚°ì˜ ê²°ê³¼ë¥¼ ì´ìš©í•´ Edgesë¥¼ ê²€ì¶œ

    #image ë‚´ì— ì›ì„ ì°¾ê³  roië¥¼ ê·¸ë ¤ë„£ëŠ” ì½”ë“œ...
    #circles = cv2.HoughCircles(gradient,method=cv2.HOUGH_GRADIENT,dp=1,minDist=50,param2=15) # í—ˆí”„ë³€í™˜ì„ ì´ìš©í•´ imageë‚´ì˜ ì›ì„ ì°¾ìŒ
    #print('circles.shape',circles.shape) #ì°¾ì€ ì›ì˜ ì¢Œí‘œë¥¼ ì¶œë ¥

    ###2ë²ˆì§¸
    mode = cv2.RETR_EXTERNAL #ì™¸ê³½ì„ ë§Œ ê·¸ë¦¬ëŠ” ëª¨ë“œ ê·¼ë° frameì„ ë²—ì–´ë‚˜ë©´ 2ê°œì˜ ì™¸ê³½ì„ ì„ ê·¸ë ¸ìŒ...ì—¬íŠ¼ ì˜ì—­ê²€ì¶œì—ëŠ” ë¬¸ì œê°€ ì—†ì—ˆìŒ.
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(gradient,mode,method) 
    x,y,width,height = cv2.boundingRect(contours[0]) 
    cv2.rectangle(frame,(x,y),(x+width,y+height),(0,0,255),3) 
    text = 'ROI'
    org = (x,y-10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,text,org,font,1,(0,0,255),3)
    # ì°¾ì€ ROI ì˜ì—­ì—ë§Œ cornersë¥¼ ì°¾ì•„ì¤€ë‹¤...
    ##????ì°¾ì€ ROI ì˜ì—­ë§Œì„ ë½‘ì•„ë‚´ëŠ” ì½”ë“œ...
    roi = gray[y : y + height, x : x + width]## ì´ê±°ë¥¼ ì–´ë–»ê²Œ ìˆ˜ì •???????????????????????????????
    K = 30  # cornerì˜ ê°œìˆ˜ë¥¼ ìµœëŒ€ 30ê°œë¡œ ì„¤ì •
    corners = cv2.goodFeaturesToTrack(roi, maxCorners=K, qualityLevel=0.05, minDistance=10, corners=None,mask=None)
    print('corners', corners)  # ì½”ë„ˆ ì¢Œí‘œ í”„ë¦°íŠ¸(ì‹¤ìˆ˜ë¡œ ì¢Œí‘œê°’ì´ ì°¾ì•„ì§)
    count = 0  # ê²€ì¶œí•œ ì½”ë„ˆì˜ ê°œìˆ˜ë¥¼ ì¹´ìš´íŠ¸í•˜ê¸° ìœ„í•œ ë³€ìˆ˜ ì„ ì–¸+ì´ˆê¸°í™”
    corners = corners.reshape(-1, 2)  # ì´ êµ¬ë¬¸ì€ ì™œ í•´ì£¼ëŠ”ì§€ ìž˜ ëª¨ë¥´ê² ìŒ...
    for a, b in corners:
        cv2.circle(frame, (int(a+x), int(b+y)), 5, (0, 0, 255), -1)  # ê²€ì¶œí•œ ì½”ë„ˆì˜ ê° ì¢Œí‘œì— ì ‘ê·¼í•´ ì›ì„ ê·¸ë¦¬ê¸°, ì¢Œí‘œ ì§€ì •ì—ì„œ ì¡°ê¸ˆ ìƒê°ì„ ë§Žì´ í–ˆìœ¼ë‚˜ í•´ê²°...
        count += 1  # ì›ì„ ê·¸ë¦´ ë•Œë§ˆë‹¤ countì˜ ê°œìˆ˜ë¥¼ í•˜ë‚˜ì”© ê°€ì‚°(í˜„ìž¬ ì½”ë„ˆì ì˜ ìˆ˜ë¥¼ ì¹´ìš´íŠ¸)
    print('count', count)  # ì¹´ìš´íŠ¸ í™•ì¸ìš©...

    if temp is not None:
        count_temp = 0  # trakingì— ì„±ê³µí•œ corner ì˜ ê°œìˆ˜
        for i in range(0, corners.shape[0]):
            for j in range(0, corners.shape[1]):
                if frame[int(i), int(j), 2] == temp[int(i), int(j), 2]:
                    count_temp += 1  # trakingì— ì„±ê³µí•œ cornerì˜ ê°œìˆ˜ë¥¼ count
                    print('count_temp', count_temp)
                else:
                    break
        org = 50, 50
        # if count > 0.55 * count_temp:
        if count > 0.5 * count_temp:
            text = 'traking success'
            org = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, org, font, 1, (255, 0, 0), 2)
        else:
            text1 = 'traking fail'
            org1 = (50, 50)
            font1 = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text1, org1, font1, 1, (255, 0, 0), 2)


    cv2.imshow('dst', dst)  # ì»¬ëŸ¬ë²”ìœ„ì— ë”°ë¥¸ ì˜ì—­ë¶„í•  í™”ë©´ í‘œì‹œ
    cv2.imshow('frame',frame) # ìµœì¢… frame
    cv2.imshow('closing', closing)  #open,close í›„ì˜ ì˜ìƒ
    cv2.imshow('gradient',gradient )  #ëª¨í´ë¡œì§€ì—°ì‚° í›„ edgesë§Œ ì¶”ì¶œ...
    temp = frame.copy()  # cornersì˜ ì¢Œí‘œê°’ì„ ë³µì‚¬í•´ì˜´...

    key = cv2.waitKey(25)
    if key == 27:
        break
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
