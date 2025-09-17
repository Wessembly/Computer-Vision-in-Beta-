import cv2, sys, time

cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("no camera"); sys.exit(2)

prev = None
AREA_MIN = 600     # lower if boxes don't appear
THRESH    = 20     # lower = more sensitive

while True:
    ok, frame = cap.read()
    if not ok: time.sleep(0.02); continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    if prev is None:
        prev = gray
        cv2.imshow("motion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    delta = cv2.absdiff(prev, gray)
    _, mask = cv2.threshold(delta, THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        if cv2.contourArea(c) < AREA_MIN: continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

    cv2.imshow("motion", frame)
    prev = gray
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()

