from ultralytics import YOLO
import cv2, time

model = YOLO("yolov8n.pt")           # 軽量モデル
cap = cv2.VideoCapture(0)            # カメラ
skip = 2                             # 間引き: 2なら 1/2 で推論
i = 0

while True:
    ok, frame = cap.read()
    if not ok: break

    # 入力サイズを下げる（例: 480pxに短辺フィット）
    h, w = frame.shape[:2]
    scale = 480 / min(h, w)
    frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))

    if i % skip == 0:
        # stream=True でI/Oと推論をパイプライン化
        results = model.predict(source=frame_small, imgsz=320, verbose=False)
        annotated = results[0].plot()
    i += 1

    cv2.imshow("det", annotated if i % skip == 0 else frame_small)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break
cap.release(); cv2.destroyAllWindows()

