import cv2

# 画像を読み込み
img = cv2.imread("room_cat.png")
h, w = img.shape[:2]

# Selective Search のセットアップ
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()  # 高速モード（粗め）
# ss.switchToSelectiveSearchQuality()  # 高品質モード（遅い）

# 候補矩形を生成
rects = ss.process()
print("候補数:", len(rects))

# 上位100個だけ描画してみる
out = img.copy()
for i, (x, y, w, h) in enumerate(rects[:100]):
    cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 結果を表示
cv2.imshow("Selective Search Result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

