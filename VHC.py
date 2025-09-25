import cv2
import time
import numpy as np
import math

import HandTrackingModule as htm 

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class VolumeController:
    
    def __init__(self, wCam=640, hCam=480, camIndex=0, detectionCon=1):
        self.wCam = wCam
        self.hCam = hCam
        self.camIndex = camIndex
        self.pTime = 0
        self.detector = htm.handDetector(detectionCon=detectionCon)
        self.cap = None

        self._setup_volume_control()
        
        self.volBar = 400
        self.volPer = 0

    def _setup_volume_control(self):
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            volRange = self.volume.GetVolumeRange()
            self.minVol = volRange[0]
            self.maxVol = volRange[1]
            print(f"Volume Range: {self.minVol} to {self.maxVol}")
        except Exception as e:
            print(f"Error setting up PyCAW: {e}")
            self.volume = None
            self.minVol = -65.0
            self.maxVol = 0.0

    
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.camIndex)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera with index {self.camIndex}")
            
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        return False

    
    def _draw_volume_bar(self, img):
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(self.volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(self.volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        return img

    def _update_fps(self, img):
        cTime = time.time()
        time_diff = cTime - self.pTime
        fps = 1 / time_diff if time_diff > 0 else 0
        self.pTime = cTime
        
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        return img

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to read frame.")
                break
                
            img = self.detector.findHands(img)
            lmList, _ = self.detector.findPosition(img, draw=False)
            
            if len(lmList) > 8:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                length = math.hypot(x2 - x1, y2 - y1)
                
                vol = np.interp(length, [50, 300], [self.minVol, self.maxVol])
                self.volBar = np.interp(length, [50, 300], [400, 150])
                self.volPer = np.interp(length, [50, 300], [0, 100])
                
                if self.volume and abs(self.volume.GetMasterVolumeLevel() - vol) > 1:
                    self.volume.SetMasterVolumeLevel(vol, None)

                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                color = (255, 0, 255)
                if length < 50:
                    color = (0, 255, 0)
                cv2.circle(img, (cx, cy), 15, color, cv2.FILLED)

            img = self._draw_volume_bar(img)
            img = self._update_fps(img)

            cv2.imshow("Volume Controller", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main():
    try:
        with VolumeController(wCam=640, hCam=480, detectionCon=0.85) as controller:
            controller.run()
            
    except IOError as e:
        print(f"System Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()