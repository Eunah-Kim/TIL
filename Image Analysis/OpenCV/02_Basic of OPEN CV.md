# 1. Basic of OPEN CV 

## 1.1 기본 입출력

### 이미지, 비디오 입출력

> 1. img를 출력해보면 이미지의 화소데이터가 출력된다. 
> 2. img = cv2.imread(file_name [, mode_flag]) : 파일으로부터 이미지 읽기 
>     * file_name : 이미지경로, 문자열 
>     * mode_flag = cv2.IMREAD_COLOR : 읽기모드지정
>         * cv2.IMREAD_COLOR : **컬러(BGR)스케일**로 읽기, 기본 값 
>         * cv2.IMREAD_UNCHANGED : 파일 그대로 읽기 
>         * **cv2.IMREAD_GRAYSCALE : 그레이(흑백) 스케일로 읽기** 
> 3. cv2.imshow(title, image) : 특정한 이미지를 화면에 출력
>     * title : 윈도우 창의 제목
>     * image : 출력할 이미지 객체
> 4. cv2.waitKey(time)
>     * time : 입력 대기 시간 (무한대기 : 0) 
>     * 사용자가 어떤키를 입력했을 때 대기하며 입력했을 때 Ascii Code(esc:27) 반환
>     * ()의 경우 아무키나 입력해도 창 닫힘
> 5. cv2.destoryAllWindow() : 화면의 모든 윈도우를 닫는 함수 



#### 이미지 읽기

**새 창에 이미지 띄우기**

```python
import cv2

img_file = 'img/actor.jpg'  #이미지 경로
img = cv2.imread(img_file)  #변수할당

cv2.imshow('IMG', img) # 'IMG'창이름으로 화면표시
cv2.waitKey() 
cv2.destroyAllWindows()
```

**jupyter notebook에 이미지 바로 나타내기**

```python
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('img/girl_face.jpg')

plt.axis('off')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

**그레이스케일로 읽기**

```python
import cv2

img_file = 'img/image.jpg'
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

cv2.imshow('IMG', img)
cv2.waitKey()
cv2.destroyAllWindows()
```



#### 동영상 파일 읽기

```python
import cv2

video_file = 'img/big_buck.avi' #파일경로

cap = cv2.VideoCapture(video_file) #객체

if cap.isOpened(): #객체 초기화 확인
    while True:
        ret, img = cap.read() #프레임 읽기
        if ret:
            cv2.imshow(video_file, img) #프레임 표시
            cv2.waitKey(15)
        else:
            break
        
    print('동영상을 열 수 없습니다.')
cap.release()
cv2.destroyAllWindows()
```



#### 비디오 파일의 프레임 간 이동

>* cap.set(id, value) : 프로퍼티 변경
>* cap.get(id) : 프로퍼티 확인 
>* cv2.CAP_PROP_POS_FRAMES : 현재프레임의 개수
```python
import cv2

capture = cv2.VideoCapture('img/big_buck.avi')
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame count:', frame_count)

#첫번째 프레임을 가져온다
capture, frame = capture.read()
cv2.imshow('frame0', frame)

#100번째 프레임 가져오기
capture = cv2.VideoCapture('img/big_buck.avi')
capture.set(cv2.CAP_PROP_POS_FRAMES, 100)
capture, frame = capture.read()
cv2.imshow('frame100', frame)

cv2.waitKey()
cv2.destroyAllWindows()
```



### 1.2 도형 그리기

#### 간단한 사각형 그리기

>* cv2.rectangle(img, start, end, color[, thickness, lineType]: 사각형 그리기 
>    * img : 그림 그릴 대상 이미지, NumPy 배열
>    * start : d사각형 시작 꼭짓점 (x,y)
>    * end : 사각형 끝 꼭짓점( x, y)
>    * color : 색상 (BGR)
>    * thickness : 선 두께 
>        * -1 : 채우기 
>    * lineType : 선타입, cv2.line()과 동일 

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = np.full((512, 512, 3), 255, np.uint8)  # (512, 512) size의 흰 이미지 만들기
image = cv2.rectangle(image, (20, 20), (255, 255), (0,0,255), 5)

plt.imshow(image)
plt.show
```



