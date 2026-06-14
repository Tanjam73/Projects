from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

PARKING_SLOTS = [
    [(50,100),(180,100),(180,250),(50,250)],
    [(220,100),(350,100),(350,250),(220,250)],
    [(390,100),(520,100),(520,250),(390,250)],
    [(560,100),(690,100),(690,250),(560,250)]
]

VEHICLE_CLASSES = {
    2,
    3,
    5,
    7
}

cap = cv2.VideoCapture("street.mp4")

def bbox_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2,(y1+y2)//2)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    occupied = [False]*len(PARKING_SLOTS)

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            if cls not in VEHICLE_CLASSES:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            cx,cy = bbox_center((x1,y1,x2,y2))

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (255,0,0),
                2
            )

            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)

            for idx,slot in enumerate(PARKING_SLOTS):

                polygon = np.array(slot,np.int32)

                inside = cv2.pointPolygonTest(
                    polygon,
                    (cx,cy),
                    False
                )

                if inside >= 0:
                    occupied[idx] = True

    free_count = 0

    for idx,slot in enumerate(PARKING_SLOTS):

        polygon = np.array(slot,np.int32)

        if occupied[idx]:
            color = (0,0,255)
        else:
            color = (0,255,0)
            free_count += 1

        cv2.polylines(
            frame,
            [polygon],
            True,
            color,
            3
        )

        x = polygon[:,0].min()
        y = polygon[:,1].min()

        status = "Occupied" if occupied[idx] else "Free"

        cv2.putText(
            frame,
            f"Slot {idx+1}: {status}",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    cv2.putText(
        frame,
        f"Free Slots: {free_count}/{len(PARKING_SLOTS)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.imshow("Curbside Parking Detection",frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
