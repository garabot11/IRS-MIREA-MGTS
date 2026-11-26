# Контур · 3D-карта и локализация Unitree Go2

Решение кейса МГТС в ROS 2. Первый обход Go2 создаёт измеренную воксельную карту помещения. Во втором обходе робот локализуется в её системе координат и на ходу обновляет слой новых препятствий и маршрут.

- [Данные организаторов](https://github.com/ivr0ma/hackathon/tree/ecbeeb681f7fe1e274402623da27444f63a4c61a/hackathon)
- [Измеренное облако и реконструкция стен](results/map)
- [Контрольные суммы исходных данных](verification/input-provenance.json)

Шаг вокселя 7 см. Результат первого обхода содержит 98 601 точку. Проверку опоры на лидарные измерения прошёл 10 351 треугольник стен.

## Воспроизведение результата

[Установка, NVIDIA Container Toolkit, CUDA, запуск ROS 2 и MuJoCo с RViz](docs/Запуск-и-GPU.md)

На Ubuntu с Docker и ROS 2 Jazzy выполните из корня проекта:

```bash
./run-contest fetch
./run-contest build
./run-contest gpu
./run-contest map
./run-contest replay
./run-contest audit
./run-contest show
```

Первый bag даёт базовую карту, второй воспроизводится в ROS 2 со скоростью 1×. В начальном режиме seeded используется стартовая одометрия. Эталонная траектория для второго обхода организаторами не предоставлена. В MuJoCo измеренная карта служит средой для повторяемых заездов с движущимися Go2 и G1.

В каталоге `output/official` появятся ROS 2 bag (MCAP), покадровые замеры и обновлённый слой карты. По умолчанию `run-contest` читает `data/` и пишет в `output/`. Для независимого повтора выберите новый путь через `CONTEST_OUTPUT`.

## Результаты для проверки

- [Выходной bag ROS 2 в формате MCAP](results/localization/rosbag2) и [покадровые замеры](results/localization/frames.csv). Скачайте оба файла каталога bag и откройте его командой `ros2 bag info results/localization/rosbag2`.
- [Время локализации и доля успешно обработанных сообщений](results/localization/timing.json).
- [Заезды Go2 и G1 в MuJoCo, контакты и расстояния до цели](results/simulation).
- [Измерения CUDA и аппаратного рендеринга EGL](verification/gpu.json).

Для просмотра выходного bag в RViz выполните в двух терминалах с загруженным окружением ROS 2 команды `rviz2 -d hackathon_solution/rviz/top.rviz` и `ros2 bag play results/localization/rosbag2 --clock`.

[Скачать одним файлом .bag](results/localization/localization.mcap.bag). Это те же байты MCAP из выходного bag ROS 2 Jazzy. Формат явно задаётся командой `ros2 bag info -s mcap results/localization/localization.mcap.bag`.

[Независимое чтение опубликованной записи](results/localization/audit.json) подтвердило 1 525 входных сканов, 1 513 опубликованных поз и 205 планов. Время от доступности входа до результата составило 9,57 мс по медиане и 21,52 мс на 95-м процентиле. В 1 523 из 1 524 измеримых интервалов результат появился до следующего скана.
