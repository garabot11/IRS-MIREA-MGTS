"""Fetch only the two pinned upstream archives and verify all extracted members."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'verification/input-provenance.json').read_text())
REVISION = 'ecbeeb681f7fe1e274402623da27444f63a4c61a'
URL = f'https://media.githubusercontent.com/media/ivr0ma/hackathon/{REVISION}/hackathon/ros2_ws/'

def check(path, expected):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'Checksum mismatch: {path.name}')

def run(action, root, name):
    spec = MANIFEST[name]
    dest = root / name
    if action == 'fetch' and not dest.exists():
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            tmp = Path(tmp)
            archive = tmp / 'download.zip'
            with urlopen(URL + name + '.zip', timeout=120) as src, archive.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            check(archive, spec['archive_sha256'])
            stage = tmp / name
            stage.mkdir()
            with zipfile.ZipFile(archive) as zipped:
                for member, expected in spec['files'].items():
                    entries = [i for i in zipped.infolist() if Path(i.filename).name == member]
                    if len(entries) != 1:
                        raise ValueError(f'Ambiguous or absent archive member: {member}')
                    with zipped.open(entries[0]) as src, (stage / member).open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    check(stage / member, expected)
            stage.rename(dest)
    for member, expected in spec['files'].items():
        check(dest / member, expected)
    print(f'{name}: SHA-256 OK')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fetch','verify'])
    parser.add_argument('root', type=Path)
    parser.add_argument('kind', choices=['mapping','localization'], nargs='?')
    args = parser.parse_args()
    for name in ([f'bag_for_{args.kind}'] if args.kind else MANIFEST):
        run(args.action, args.root, name)
