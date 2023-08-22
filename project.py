#Kullanacağımız kütüphaneyi çalışmamıza dahil edelim.
import cv2

#Kullanacağımız videoyu çalışmamıza dahil edelim.
cap = cv2.VideoCapture('C:\Opencv_haarcascade\/test_videos\People.avi')
#Kullanacağımız cascade dosyalarını çalışmamıza dahil edelim
human_cascade = cv2.CascadeClassifier('C:\Opencv_haarcascade\haar_cascade\/fullbody.xml')

#Sonsuz bir döngü ile her kareyi(frame) tek tek inceleyelim.
while (True):
#Her kareyi tek tek okuyalım.
    ret, frame = cap.read()
    frame = cv2.resize(frame, (600, 600))
#Haar-like özellikleri kolay algılayabilmek için her bir8 kareyi gri tonlara çevirelim.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#Şimdi de cascade dosyamızı kullanarak her bir kare üzerindeki yüzlerin koordinarlarını bulalım.
    humans = human_cascade.detectMultiScale(gray, 1.9, 1)

#"bodies" değişkeninde tuttuğumuz koordinatları kullanarak yüzleri dikdörtgen içerisine alalım.
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#İşlediğimiz kareleri görelim.
    cv2.imshow('frame', frame)
#Programı kapatacak kodu yazalım.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Son olarak videoyu serbest bırakalım.
cap.release()
cv2.destroyAllWindows()
""" Webcam den görüntü almak istiyorsak parantez içinde video yolu silinip 0 yazılır.Harici bir kameradan görüntü almak istiyorsak bilgisa- 
 yara bağlanıp 1 yazılır."""
