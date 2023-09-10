Dober Team решение кейса Меркатор (Урбаномика) всероссийский финал Цифровой Прорыв 2023
==============

[//]: # (Image References)
[image1]: ./misc/pit_detected.jpg "Test output"
[image2]: ./misc/ground_segmented.jpg "Test output"
[image3]: ./misc/rqt_graph.jpg "Test output"

![test output][image1]

##### Визуализация работы модуля детекции дефектов дорожного покрытия на видео

![test output][image2]

##### Визуализация работы модуля сегментации земной поверхности (белое — земля, остально цветное)

![test output][image3]

##### Визуализация графа модулей и топиков ROS2

## Описание

### Модуль обнаружения дорожных дефектов

В качестве детектора дорожных дефектов была выбрана готовая модель для Zero-Shot Detection [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), т.к. размеченного обучающего датасета не было, как и ресурсов чтобы быстро его разметить.

Для ускорения обработки мы обрабатываем нейросетью 1 кадр за секунду. Скорость обработки кадра на сервере с GPU Nvidia GTX1060 6GB как раз около 1 кадра в секунду.

Между детекцией и трекингом добавлен этап Non-maxima suppression, т.к. детектор часто выдает несколько боксов на одной позиции.

В качестве трекера используется Deep-SORT.

Уникальной особенностью модуля нейросетевой детекции является инвариантность к классам детекции, так и к ракурсам съемки. При помощи текстового описания требуемого класса (на английском языке) можно детектировать объекты, под которые нейросеть не обучалась.

Протестирована совместимость с python3.10, Ubuntu 22.04 и ROS2 Humble.

### Модуль сегментации земной поверхности в облаке точек

Для сегментации земли использовался подход [TRAVEL: Traversable Ground and Above-Ground Object Segmentation using Graph Representation for 3D LiDAR Scans](https://arxiv.org/abs/2206.03190)

Модуль выполнен в виде пакета для ROS2 Humble и находится в папке src/travel.

Он подписывается на топик /points с облаком точек от лидара робота Пикселя и выдаёт 2 облака точек в топики /ground и /nonground для точек земной плоскости и остальных точек соответственно.

## Установка сервиса поиска дорожных дефектов

```bash
git clone https://github.com/alex4men/dober_pits.git
cd dober_pits
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Тестирование сервиса поиска дорожных дефектов

В файле pipeline.py можно задать путь к входному и выходному видео, а так же список классов для поиска в самом конце файла.

```bash
cd src
python pipeline.py
```

## Деплой в продакшн сервиса поиска дорожных дефектов

На голом железе

```bash
uvicorn service:app --host 0.0.0.0 --port 3000
```

Либо в Докере

```bash
docker build -t dober_ml_img .
docker run -d --name dober_ml_cont -p 8086:8086 dober_ml_img
```

## Установка сервиса сегментации земной поверхности

Установить Ubuntu 22, ROS2 Humble.

```bash
source opt/ros/humble/setup.bash
cd dober_pits
colcon build --symlink-install
```

## Запуск модуля сегментации земной поверхности

```bash
cd dober_pits
source install/setup.bash
ros2 run travel travel_exe --ros-args --params-file src/travel/config/travel.param.yaml
```

Если в ROSbag нет трансляции топика /tf и /tfstatic, то для работы модуля необходимо запустить команду:

```bash
ros2 run tf2_ros static_transform_publisher "0" "0" "0" "0" "0" "0" "map" "laser_link"
```
