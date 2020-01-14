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

#### 간단한 사각형 그리기 1

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

#### 간단한 사각형 그리기 1

```python
import cv2

img = cv2.imread('img/blank_500.jpg')

cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)

cv2.imshow('rectanvle', img)
```



#### 다각형 그리기

> * cv2.polylines(img, points, isClosed, color[, thickness, lineType]): 다각형 그리기 
>
>     *img : 그림 그릴 대상 이미지 
>     
>     * points : 꼭짓점 좌표, Numpy 배열 리스트 
>     * isClosed: 닫힌 도형 여부, True/False 
>     * color : 색상(BGR)
>     * thickness : 선 두께
>     * lineType : 선 타입, cv2.line()과 동일

```python
import cv2

image = np.full((512, 512, 3), 255, np.uint8)
points = np.array([[5, 5], [128, 312], [483, 444], [400, 150]])
image = cv2.polylines(image, [points], True, (0, 0, 255), 4)

plt.imshow(image)
plt.show
```



#### 원 그리기

> * cv2.circle(img, center, radius, color[, thickness, lineType]) : 원 그리기
>     * img : 그림 대상 이미지
>     * center : 원점 좌표 (x,y)
>     * radius : 원의 반지름 
>     * color : 색상 (BGR)
>     * thickness : 선 두께 (-1 : 채우기)
>     * lineType : 선 타입, cv2.line()과 동일

```python
import cv2

image = np.full((512, 512, 3), 255, np.uint8)
image = cv2.circle(image, (255, 255), 100, (255, 200, 0), -1)

plt.imshow(image)
plt.show
```



#### 텍스트 삽입

> * cv2.putText(image, text, position, font_type, font_scale, color) : 하나의 텍스트를 그리는 함수 
>      - position : 텍스트가 출력될 위치 
>      - font_type : 글씨체 
>      - font_scale: 글씨 크기 가중치

```python
import cv2

image = np.full((512, 512, 3), 255, np.uint8)
image = cv2.putText(image, 'Hello World', (0, 200), cv2.FONT_ITALIC, 2, (0,0, 0))

cv2.imshow('text', image)
```



### 1.3 창 관리

> * cv2.namedWindow(title [, option]) : 이름을 갖는 창 열기 
>     * title : 창이름, 제목 줄에 표시
>     * option : 창옵션 
>         * cv2.WINDOW_NORMAL:임의의크기, 창 크기 조정 가능 
>         * cv2.WINDOW_AUTOSIZE : 이미지와 같은 크기, 창 크기 재조정 불가능 
> * cv2.moveWindow(title, x좌표 , y좌표) : 창위치 이동 
> * cv2.resizeWindow(title, width, height) : 창 크기 변경 
> * cv2.destroyWindow(title) : 창 닫기 
> * cv2.destroyAllWindows(): 열린 모든 창 닫기

```python
import cv2

img = cv2.imread('img/boy_face.jpg')
img_gray = cv2.imread('img/boy_face.jpg', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('origin')
cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
cv2.imshow('origin', img)
cv2.imshow('gray', img_gray)

#창 위치 변경
cv2.moveWindow('origin', 0, 0)
cv2.moveWindow('gray', 100, 100)

#창 크기 변경
cv2.waitKey()
cv2.resizeWindow('origin', 200, 200)
cv2.resizeWindow('gray', 100, 100)

cv2.waitKey()
cv2.destroyAllWindows()
```



