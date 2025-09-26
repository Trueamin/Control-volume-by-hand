# کتابخانه‌های مورد نیاز
import cv2  # پردازش تصویر و کار با وبکم
import time  # محاسبه فریم بر ثانیه
import numpy as np  # محاسبات عددی
import HandTrackingModule as htm  # ماژول تشخیص دست
import math  # محاسبات ریاضی
from ctypes import cast, POINTER  # کنترل صدا در ویندوز
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # کنترل صدا

# تنظیمات ابعاد تصویر
wCam, hCam = 640, 480  # عرض و ارتفاع تصویر

# تنظیمات دوربین
cap = cv2.VideoCapture(0)  # اتصال به وبکم پیش‌فرض
cap.set(3, wCam)  # تنظیم عرض تصویر
cap.set(4, hCam)  # تنظیم ارتفاع تصویر
pTime = 0  # زمان قبلی برای محاسبه FPS

# ایجاد شیء تشخیص دست با دقت بالا
detector = htm.handDetector(detectionCon=1)

# بخش کنترل صدا در ویندوز
devices = AudioUtilities.GetSpeakers()  # دریافت دستگاه خروجی صدا
interface = devices.Activate(          # ایجاد رابط کنترل صدا
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))  # تبدیل به شیء قابل استفاده
volRange = volume.GetVolumeRange()  # دریافت محدوده صدا (معمولاً [-65, 0])
minVol = volRange[0]  # حداقل حجم صدا
maxVol = volRange[1]  # حداکثر حجم صدا
vol = 0  # مقدار فعلی صدا
volBar = 400  # موقعیت اولیه نوار صدا
volPer = 0  # درصد صدا

# حلقه اصلی برنامه
while True:
    success, img = cap.read()  # دریافت فریم از دوربین
    if not success:  # اگر دریافت فریم ناموفق بود
        break  # خروج از حلقه
    
    # تشخیص دست در تصویر
    img = detector.findHands(img)
    # دریافت موقعیت نقاط دست (بدون رسم)
    lmList, _ = detector.findPosition(img, draw=False)
    
    # اگر دست تشخیص داده شد
    if len(lmList) != 0:
        # بررسی وجود نقاط کلیدی (نوک انگشتان)
        if len(lmList) > 8:  # نیاز به حداقل 9 نقطه داریم
            # مختصات نوک انگشت شست (نقطه 4) و انگشت اشاره (نقطه 8)
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            # محاسبه نقطه میانی
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # رسم نقاط و خط بین انگشتان
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # شست
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)  # اشاره
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # خط اتصال
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # نقطه میانی

            # محاسبه فاصله بین دو انگشت
            length = math.hypot(x2 - x1, y2 - y1)
            
            # تبدیل فاصله به محدوده صدا
            # محدوده دست: 50-300 پیکسل → محدوده صدا: minVol تا maxVol
            vol = np.interp(length, [50, 300], [minVol, maxVol])
            volBar = np.interp(length, [50, 300], [400, 150])  # برای نمایش بصری
            volPer = np.interp(length, [50, 300], [0, 100])  # درصد صدا
            
            # تنظیم صدا فقط در صورت تغییر محسوس (برای بهینه‌سازی)
            if abs(volume.GetMasterVolumeLevel() - vol) > 1:
                volume.SetMasterVolumeLevel(vol, None)

            # تغییر رنگ نقطه میانی وقتی فاصله کم است (حالت بسته)
            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # رسم نوار حجم صدا
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)  # قاب نوار
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)  # سطح صدا
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)  # نمایش درصد صدا

    # محاسبه و نمایش FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Img", img)  # نمایش تصویر
    if cv2.waitKey(1) & 0xFF == ord('q'):  # خروج با فشار کلید q
        break

# آزادسازی منابع
cap.release()  # آزاد کردن دوربین
cv2.destroyAllWindows()  # بستن تمام پنجره‌ها
