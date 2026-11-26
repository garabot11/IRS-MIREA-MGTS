# Запуск на Ubuntu и проверка NVIDIA GPU

Из корня публичной поставки можно построить карту по первому обходу организаторов, локализовать Go2 на втором и запустить физическую симуляцию двух Go2 и гуманоидного G1. Для проверки самого GPU записи не требуются. Начните с диагностики оборудования, затем соберите два образа и выполните контрольный расчёт.

## Что использует видеокарту

**Для полного графического режима и проверки NanoDet на CUDA нужна NVIDIA GPU с рабочим драйвером и NVIDIA Container Toolkit.** CPU-режим подходит для картографирования, локализации и проверки симуляции с программной графикой. Скорость окон и вывода модели зависит от выбранного режима.

| Этап | Где выполняется | Как проверить |
| --- | --- | --- |
| Локальный детектор NanoDet | ONNX Runtime с CUDA на NVIDIA. Веса находятся в образе картографирования, в файле `/project/inspection/models/nanodet.onnx` | Запустить `tools/kontur-gpu verify --mode gpu` и проверить результат вычисления модели на CUDA |
| Кадр MuJoCo без рабочего стола | NVIDIA OpenGL через EGL | Та же проверка создаёт графический контекст и рисует кадр. В результате нужен рендерер NVIDIA |
| Окна MuJoCo и RViz | OpenGL в графической сессии, для NVIDIA используется GLX | На рабочем столе открыть оба окна и проверить графический драйвер в этой сессии |
| Геометрия карты и стен, ROS 2, локализация и планирование | CPU | Проверить выходной PLY, покадровые метки, результаты аудита записи |
| Динамика MuJoCo и геометрические лучи лидара, опциональная временная модель движения NumPy | CPU | Телеметрия симуляции содержит положение, команды, контакты и модельное время |

Проверка NanoDet характеризует локальный детектор. В официальных записях организаторов есть лидар, одометрия и IMU. Показатели скорости картографирования и локализации относятся к обработке этих данных.

## Подготовка хоста

Команды рассчитаны на Bash и Ubuntu 24.04. На контрольной машине использованы Ubuntu 24.04.4 LTS, GeForce RTX 5060 Laptop GPU, драйвер 580.178.04, NVIDIA Container Toolkit 1.20.1 и Docker Engine 29.7.2. Эти версии описывают стенд. Для нового компьютера выберите пакеты, поддерживающие его видеокарту. Docker Engine должен быть доступен пользователю, который запускает проект. [Установка Docker Engine на Ubuntu](https://docs.docker.com/engine/install/ubuntu/) · [Драйвер и NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
tools/kontur-gpu doctor
nvidia-smi --query-gpu=name,driver_version --format=csv
docker version
nvidia-ctk --version
```

До первой сборки `doctor` также сообщит об отсутствующих локальных образах и завершится с кодом 1. После сборки повторите проверку. В выводе `nvidia-smi` должны появиться имя устройства и версия драйвера, в `docker version` — версии клиента и движка. Если драйвер или Container Toolkit нужно установить, сначала посмотрите доступные варианты и план действий.

```bash
tools/kontur-gpu install --help
tools/kontur-gpu install
```

После просмотра плана выполните `sudo tools/kontur-gpu install --apply`. При отсутствии рабочего драйвера добавьте `--with-driver` и к просмотру плана, и к применению. Установщик рассчитан на Ubuntu x86_64 с системным Docker. Перезапуск службы после изменения конфигурации выполняет администратор. Настройка системной службы Docker может потребовать перезапуска и прервать запущенные контейнеры. При ручной настройке по инструкции NVIDIA после установки toolkit выполняют `sudo nvidia-ctk runtime configure --runtime=docker` и затем `sudo systemctl restart docker`. После изменения драйвера или runtime повторите диагностику. [Настройка Docker с NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)

Для воспроизведения ROS 2 и показа окон на хосте нужен ROS 2 Jazzy. Подключите официальный репозиторий ROS для Ubuntu Noble, затем установите пакеты и откройте Bash с окружением Jazzy. [Установка ROS 2 Jazzy на Ubuntu](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

```bash
sudo apt install ros-jazzy-desktop ros-jazzy-rosbag2-storage-mcap \
  python3-numpy python3-scipy python3-opencv python3-yaml
source /opt/ros/jazzy/setup.bash
ros2 pkg prefix rosbag2_storage_mcap
ros2 pkg prefix rviz2
```

## Сборка и контрольный расчёт

```bash
./run-contest build
tools/kontur-gpu verify --mode gpu
```

Сборка создаёт образы `kontur-mapping:gpu` и `kontur-physics:local`. Первый содержит локальный детектор и обработку измерений, второй — MuJoCo, модели роботов и графические библиотеки. Если вы собираете образы вручную, используйте корень публичной поставки как контекст.

```bash
docker build -f docker/mapping.Dockerfile -t kontur-mapping:gpu .
docker build -f docker/physics.Dockerfile -t kontur-physics:local .
```

Контрольный запуск без NVIDIA выполняется командой `tools/kontur-gpu verify --mode cpu` и использует ONNX Runtime на CPU и программный EGL. Оба режима используют синтетический вход и локальные образы. Для проверки других тегов задайте оба образа явно. CUDA и графический рендер проверяются независимо. При программном рендеринге может появиться изображение, но строка `llvmpipe` указывает на выполнение графики на CPU.

```bash
tools/kontur-gpu verify --mode gpu \
  --inspection-image kontur-mapping:gpu --physics-image kontur-physics:local
```

Docker передаёт NVIDIA-устройства контейнеру через `--gpus all`. Возможность `compute` нужна для CUDA, `graphics` для OpenGL, `display` для X11, `utility` для NVML и `nvidia-smi`. В собственном Compose-файле эквивалентный запрос устройств — `gpus: all`, начиная с Compose 2.30. [Возможности драйвера NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html) · [Параметр gpus в Docker Compose](https://docs.docker.com/reference/compose-file/container-spec/#gpus)

## Официальные записи, карта и второй проход

Скачайте два обхода командой ниже. Если записи уже распакованы, поместите в `data/bag_for_mapping` и `data/bag_for_localization` соответствующие `metadata.yaml` и файлы MCAP. Если архив содержит одноимённые вложенные папки, выбирайте каталог с самим `metadata.yaml`. Входами служат только эти две записи. [Данные и описание организаторов](https://github.com/ivr0ma/hackathon)

```bash
./run-contest fetch
ros2 bag info data/bag_for_mapping
ros2 bag info data/bag_for_localization
./run-contest map
./run-contest replay
./run-contest audit
```

| Результат | Что открыть |
| --- | --- |
| `output/baseline` | Траектория и предварительная карта первого обхода |
| `output/map` | `map3d.ply` с измеренными точками, `grid.npz` с сеткой, `wall-support.json` с хешем исходного облака и качеством геометрической поддержки. При достаточном покрытии здесь появляется `wall-surface.stl` |
| `output/official` | Выходная запись ROS 2 в `rosbag2/`, строки по сканам в `frames.csv`, замеры времени в `timing.json`, измеренные дополнения в `online_delta.ply` |

Карта строится по первому обходу. Второй воспроизводится с исходной скоростью. Аудит сверяет метки времени сканов, сообщения ROS и версии карты, использованные для планирования. Для опубликованной пары записей ожидаются 1 688 сканов первого обхода и 1 525 второго. Начальная привязка занимает несколько сканов, поэтому число выданных поз может быть меньше числа входных сканов. Планировщик получает новые измерения и записывает свои решения. Физические последствия его команд проверяются в симуляторе.

## Симуляция и рабочий стол

Из терминала графической сессии X11 или XWayland проверьте доступность её дисплея, затем запустите показ.

```bash
printf '%s\n' "$DISPLAY"
./run-contest show
```

Откроются два нативных окна. В MuJoCo видны физические тела и движение приводов, в RViz — те же участники, облако измерений, цель и текущий план. Оба окна получают состояние одной симуляции. В `output/team` появятся отдельные каталоги заездов. На ноутбуке с гибридной графикой производителя OpenGL можно проверить в этом же терминале командой `glxinfo -B` из пакета `mesa-utils`. Для выбора NVIDIA при GLX используется `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia glxinfo -B`. Для контрольного кадра EGL экран не требуется.

## Если проверка остановилась

| Наблюдение | Следующая проверка |
| --- | --- |
| Хостовый `nvidia-smi` сообщает ошибку | Проверьте версию установленного драйвера и загрузку модуля ядра. После обновления загрузитесь с соответствующим ядром и снова проверьте карту |
| Хост видит GPU, а Docker сообщает об отсутствии GPU runtime | Проверьте `nvidia-ctk --version`, доступ к Docker и `docker info --format '{{json .Runtimes}}'`. Настройте runtime и повторите `tools/kontur-gpu verify --mode gpu` |
| ONNX Runtime выполняет модель на CPU или сообщает о недостающей библиотеке CUDA | Проверьте, что выбран образ `kontur-mapping:gpu`, драйвер совместим с библиотеками образа и контейнер получил GPU. Пересоберите этот образ и повторите контрольный расчёт |
| EGL работает, окно RViz или MuJoCo не открывается | Запустите показ в терминале текущей графической сессии. Проверьте `DISPLAY`, доступ X11 и рендерер GLX. Для нативных окон необходимы графические разрешения этой сессии |
| Читается пустой bag или отсутствует MCAP | Проверьте расположение `metadata.yaml` рядом с указанными в нём файлами. Для проверки состава выполните `ros2 bag info` для соответствующего каталога |
| Каталог результата уже существует | Сохраните предыдущий результат под другим именем и повторите нужный этап с новым каталогом назначения |
