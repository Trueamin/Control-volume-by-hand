import cv2
import mediapipe as mp
import time
import math

class HandDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, camIndex=0):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.camIndex = camIndex

        self.mpHands = mp.solutions.hands
        self.hands = None
        self.mpDraw = mp.solutions.drawing_utils
        
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []
        self.results = None
        self.cap = None

    def __enter__(self):
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.cap = cv2.VideoCapture(self.camIndex)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera with index {self.camIndex}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
        cv2.destroyAllWindows()
        return False 


    def _process_image(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

    def findHands(self, img, draw=True):
        self._process_image(img)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        bbox = []
        xList = []
        yList = []

        if self.results and self.results.multi_hand_landmarks and handNo < len(self.results.multi_hand_landmarks):
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if xList and yList:
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

                if draw:
                    pad = 20
                    cv2.rectangle(img, (xmin - pad, ymin - pad),
                                  (xmax + pad, ymax + pad), (0, 255, 0), 2)
        
        return self.lmList, bbox

    def fingersUp(self):
        if not self.lmList:
            return [0] * 5

        fingers = []

        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):

        if len(self.lmList) <= max(p1, p2):
            return 0, img, [0, 0, 0, 0, 0, 0]

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        length = math.hypot(x2 - x1, y2 - y1)
        
        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        
        cTime = time.time()
        
        time_diff = cTime - pTime
        if time_diff > 0:
            fps = 1 / time_diff
        else:
            fps = 0
            
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                   (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()