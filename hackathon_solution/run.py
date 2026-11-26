"""Сборка карты, причинное воспроизведение ROS 2 и проверка готовой записи."""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import yaml


SCAN = '/utlidar/cloud_base'
ODOM = '/utlidar/robot_odom'
TOPICS = {
    '/kontur/map3d': 'sensor_msgs/msg/PointCloud2',
    '/kontur/scan': 'sensor_msgs/msg/PointCloud2',
    '/kontur/pose': 'geometry_msgs/msg/PoseStamped',
    '/kontur/path': 'nav_msgs/msg/Path',
    '/kontur/status': 'std_msgs/msg/String',
    '/tf': 'tf2_msgs/msg/TFMessage',
}


def metadata(directory):
    with (directory / 'metadata.yaml').open() as stream:
        return yaml.safe_load(stream)['rosbag2_bagfile_information']


def topics(info):
    return {t['topic_metadata']['name']: (t['topic_metadata']['type'], t['message_count'])
            for t in info['topics_with_message_count']}


def scan_count(bag):
    table = topics(metadata(bag))
    for name, msg_type in [(SCAN, 'sensor_msgs/msg/PointCloud2'), (ODOM, 'nav_msgs/msg/Odometry')]:
        if name not in table or table[name][0] != msg_type or table[name][1] < 1:
            raise ValueError(f'В {bag} отсутствуют сообщения {name} типа {msg_type}')
    return table[SCAN][1]


def bag_reader(directory, info):
    import rosbag2_py as bag2
    reader = bag2.SequentialReader()
    reader.open(bag2.StorageOptions(uri=str(directory), storage_id=info['storage_identifier']),
                bag2.ConverterOptions('', ''))
    return reader


def source_stamps(source):
    import rosbag2_py as bag2
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    reader = bag_reader(source, metadata(source))
    reader.set_filter(bag2.StorageFilter(topics=[SCAN]))
    while reader.has_next():
        _, payload, _ = reader.read_next()
        stamp = deserialize_message(payload, PointCloud2).header.stamp
        yield stamp.sec * 1_000_000_000 + stamp.nanosec


def verify(output, source):
    """Compare original sensor stamps, CSV and the actual recorded output, including abstentions."""
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String

    expected = scan_count(source)

    with (output / 'frames.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with (output / 'timing.json').open() as stream:
        timing = json.load(stream)
    bag = output / 'rosbag2'
    info = metadata(bag)
    table = topics(info)
    for name, msg_type in TOPICS.items():
        if name not in table or table[name][0] != msg_type:
            raise ValueError(f'В выходном bag отсутствует {name} с типом {msg_type}')
    checked = 0
    for sensor_ns in source_stamps(source):
        if checked >= len(rows) or int(rows[checked]['sensor_ns']) != sensor_ns:
            raise ValueError(f'Скан №{checked} в CSV расходится с заголовком исходного лидара')
        checked += 1
    if checked != expected:
        raise ValueError(f'В исходном MCAP прочитано {checked} сканов, заявлено {expected}')
    reader = bag_reader(bag, info)
    counts = Counter()
    while reader.has_next():
        name, payload, time_ns = reader.read_next()
        if name == '/kontur/status':
            idx = counts[name]
            status = json.loads(deserialize_message(payload, String).data)
            if (idx >= len(rows) or int(rows[idx]['playback_ns']) != time_ns
                    or status['scan_index'] != idx or status['sensor_ns'] != int(rows[idx]['sensor_ns'])
                    or status['state'] != rows[idx]['state']):
                raise ValueError(f'Состояние скана №{idx} расходится между MCAP и CSV')
        counts[name] += 1
    if dict(counts) != {name: count for name, (_, count) in table.items() if count}:
        raise ValueError(f'Счётчики в хранилище расходятся с metadata.yaml: {counts}')
    if len(rows) != expected or counts['/kontur/status'] != expected:
        raise ValueError(f"Исходных сканов {expected}, строк CSV {len(rows)}, статусов {counts['/kontur/status']}")
    poses = sum(bool(row.get('x')) for row in rows)
    if not timing['complete'] or timing['expected_scans'] != expected or timing['input_scans'] != expected:
        raise ValueError('timing.json не подтверждает завершение обработки исходной записи')
    if poses != counts['/kontur/pose'] or poses != counts['/kontur/scan'] or poses != timing['poses']:
        raise ValueError('Число опубликованных поз расходится с CSV, timing.json или числом облаков')
    if counts['/kontur/map3d'] < 1 or (poses and (not counts['/kontur/path'] or not counts['/tf'])):
        raise ValueError('Отсутствует карта, путь или система координат для выданных поз')
    if timing['states'] != Counter(row['state'] for row in rows):
        raise ValueError('Распределение состояний расходится между CSV и timing.json')
    if any(row.get('odom_ns') and int(row['odom_ns']) > int(row['sensor_ns']) for row in rows):
        raise ValueError('В регистрации использована одометрия из будущего')
    if (int(rows[-1]['playback_ns']) - int(rows[0]['playback_ns']) < 1_000_000 and
            int(rows[-1]['sensor_ns']) - int(rows[0]['sensor_ns']) > 1_000_000_000):
        raise ValueError('Время публикации не продвигалось вместе с воспроизведением')
    return {'input_scans': expected, 'frames_csv': len(rows), 'storage': info['storage_identifier'],
            'bag_messages': dict(counts)}


@contextmanager
def process(command, env):
    proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        yield proc
    finally:
        if proc.poll() is None:
            # Let ROS close its writer before releasing the staging directory.
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def replay(args):
    bag, map_file, output = args.bag.resolve(), args.map.resolve(), args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Выходной каталог уже существует: {output}')
    if not map_file.is_file():
        raise FileNotFoundError(map_file)
    expected = scan_count(bag)
    duration = metadata(bag)['duration']['nanoseconds'] * 1e-9
    output.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'ROS_LOCALHOST_ONLY': '1'}
    with tempfile.TemporaryDirectory(prefix='.localization-', dir=output.parent) as tmp:
        work = Path(tmp)
        ready, staged = work / 'ready', work / 'result'
        node_cmd = [sys.executable, '-m', 'hackathon_solution.ros_node', '--map', str(map_file),
                    '--output', str(staged), '--start-mode', args.start_mode,
                    '--ready-file', str(ready), '--expected-scans', str(expected)]
        for flag in ("online_mapping", "motion_compensation"):
            if getattr(args, flag):
                node_cmd.append("--" + flag.replace("_", "-"))
        if args.reliable_input:
            node_cmd.append('--reliable-input')
        with process(node_cmd, env) as node:
            deadline = time.monotonic() + 60
            while not ready.exists():
                if node.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f'Нода не подтвердила готовность, код {node.poll()}')
                time.sleep(0.05)
            # Only the two subscribed sensor topics cross the playback graph.
            play = ['ros2', 'bag', 'play', str(bag), '--rate', '1', '--delay', '2',
                    '--clock', '100', '--topics', SCAN, ODOM]
            if args.reliable_input:
                # Finite controlled tests can request DDS acknowledgement for the last sample.
                qos = work / 'qos.yaml'
                sensor_qos = {'history': 'keep_last', 'depth': 128, 'reliability': 'reliable',
                              'durability': 'volatile'}
                qos.write_text(yaml.safe_dump({SCAN: sensor_qos, ODOM: sensor_qos}))
                play += ['--wait-for-all-acked', '2000', '--qos-profile-overrides-path', str(qos)]
            with process(play, env) as player:
                deadline = time.monotonic() + duration + 45
                while player.poll() is None:
                    if node.poll() not in (None, 0) or time.monotonic() > deadline:
                        raise RuntimeError(f'Воспроизведение прервано, код ноды {node.poll()}')
                    time.sleep(0.1)
                if player.returncode:
                    raise RuntimeError(f'ros2 bag play завершился с кодом {player.returncode}')
            try:
                node.wait(timeout=15)
            except subprocess.TimeoutExpired as exc:
                os.killpg(node.pid, signal.SIGINT)
                node.wait(timeout=8)
                diag = Path(tempfile.mkdtemp(prefix=output.name + '.incomplete-', dir=output.parent))
                for name in ('frames.csv', 'timing.json'):
                    if (staged / name).is_file():
                        shutil.copy(staged / name, diag / name)
                if (diag / 'frames.csv').is_file():
                    with (diag / 'frames.csv').open() as stream:
                        observed = {int(r['sensor_ns']) for r in csv.DictReader(stream)}
                    missing = [ns for ns in source_stamps(bag) if ns not in observed]
                    (diag / 'missing_sensor_ns.json').write_text(json.dumps(missing) + '\n')
                raise RuntimeError(f'Нода не подтвердила все {expected} сканов. Диагностика: {diag}') from exc
            if node.returncode:
                raise RuntimeError(f'Нода завершилась с кодом {node.returncode}')
        report = verify(staged, bag)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staged.rename(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    mapping = commands.add_parser('map', help='Собрать карту по первому обходу (среда inspection)')
    mapping.add_argument('bag', type=Path)
    mapping.add_argument('output', type=Path)
    local = commands.add_parser('replay', help='Локализовать второй обход при воспроизведении 1×')
    local.add_argument('bag', type=Path)
    local.add_argument('map', type=Path, help='Файл map3d.ply')
    local.add_argument('output', type=Path)
    local.add_argument('--start-mode', choices=['seeded', 'global'], default='seeded')
    local.add_argument('--reliable-input', action='store_true',
                       help='Дождаться подтверждения сообщений DDS при воспроизведении конечного bag')
    local.add_argument('--online-mapping', action='store_true', help='Обновлять 3D-слой и перепланировать путь в движении')
    local.add_argument('--motion-compensation', action='store_true', help='Компенсировать движение по времени каждой точки')
    check = commands.add_parser('verify', help='Сверить исходный и записанный bag, CSV и замеры')
    check.add_argument('bag', type=Path, help='Входной bag локализации')
    check.add_argument('output', type=Path, help='Каталог результатов локализации')
    args = parser.parse_args()
    if args.command == 'map':
        os.execv(sys.executable, [sys.executable, '-m', 'hackathon_solution.build_map',
                                  str(args.bag), str(args.output)])
    if args.command == 'verify':
        print(json.dumps(verify(args.output, args.bag), ensure_ascii=False, indent=2))
    else:
        # SIGTERM from a service manager must also unwind both child processes.
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
        replay(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        sys.exit(f'Ошибка: {error}')
