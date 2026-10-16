"""Map original Cars annotation IDs to the author split; preserve full image bytes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

from PIL import Image
from scipy.io import loadmat
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.experiment import write_json

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = Path('/root/datasets/fasterVitBase/raw/cars/StanfordCars-Dataset-main/stanford_cars')
ROOT = REPO / 'data/cars_author/StanfordCars_fewshot'
AUTHOR = REPO / 'data/provenance/sceg_author/StanfordCars_fewshot_splits.txt'
AUTHOR_SHA = '2677107210b2bf1c9dfe1741ddc7d6d8dad992232ff04efa24ba4676aa48052c'
PROTOCOL = 'sceg_author_cars_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_split(path=AUTHOR):
    if digest(path) != AUTHOR_SHA:
        raise ValueError('Author original changed: inspect user-supplied version; never rewrite it')
    groups = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r'\[(train|val|test)\]', line)
        if match:
            current = match[1]
            groups[current] = []
        elif line.strip().isdigit():
            groups[current].append(int(line))
    assert [len(groups[k]) for k in ('train', 'val', 'test')] == [130, 17, 49]
    ids = sum(groups.values(), [])
    assert len(ids) == len(set(ids)) == 196 and set(ids) == set(range(1, 197))
    return groups


def annotation_records(source=SOURCE):
    names = [str(x) for x in loadmat(source / 'devkit/cars_meta.mat', simplify_cells=True)['class_names']]
    assert len(names) == len(set(names)) == 196
    records = []
    for partition, annotation in [('cars_train', 'devkit/cars_train_annos.mat'),
                                  ('cars_test', 'cars_test_annos_withlabels.mat')]:
        rows = loadmat(source / annotation, simplify_cells=True)['annotations']
        filenames = [str(row['fname']) for row in rows]
        assert len(filenames) == len(set(filenames))
        assert set(filenames) == {p.name for p in (source / partition).glob('*.jpg')}
        for row in rows:
            filename, cid = str(row['fname']), int(row['class'])
            assert Path(filename).name == filename and 1 <= cid <= 196
            records.append({'identity': f'{partition}/{filename}', 'source_partition': partition,
                            'source_path': str(source / partition / filename),
                            'original_class_id': cid, 'class_name': names[cid-1]})
    assert len(records) == len({r['identity'] for r in records}) == 16185
    # Independent full-archive annotation has different image names; check only its
    # partition/class histogram, never invent a filename correspondence.
    full = loadmat(source / 'cars_annos.mat', simplify_cells=True)['annotations']
    assert Counter((int(r['test']), int(r['class'])) for r in full) == Counter(
        (int(r['source_partition'] == 'cars_test'), r['original_class_id']) for r in records)
    return records, names


def build(source=SOURCE, root=ROOT):
    started = time.perf_counter()
    prep = HERE / 'prep'
    prepared_path = prep / 'prepared.json'
    if prepared_path.exists():
        existing = json.loads(prepared_path.read_text())
        assert existing['root'] == str(root) and existing['source_root'] == str(source)
        assert digest(prep / 'samples.jsonl') == existing['sample_manifest_sha256']
        assert digest(prep / 'split.json') == existing['split_manifest_sha256']
        print(json.dumps({'status': existing['status'], 'reused': True, 'prepared': str(prepared_path)}))
        return
    groups = parse_split()
    records, names = annotation_records(source)
    membership = {cid: split for split, ids in groups.items() for cid in ids}
    class_to_idx = {s: {f'{cid:03d}': i for i, cid in enumerate(sorted(ids))} for s, ids in groups.items()}
    prep.mkdir(parents=True, exist_ok=True)
    resize = transforms.Compose([transforms.Resize(92), transforms.CenterCrop(84)])
    counts, sizes, modes = Counter(), [], Counter()
    destinations = set()
    manifest = prep / 'samples.jsonl'
    partial = manifest.with_suffix('.jsonl.part')
    with partial.open('w') as out:
        for record in records:
            cid = record['original_class_id']
            split = membership[cid]
            filename = record['source_partition'] + '__' + Path(record['source_path']).name
            relative = Path(split) / f'{cid:03d}' / filename
            assert str(relative) not in destinations
            destinations.add(str(relative))
            target, original = root / relative, Path(record['source_path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                assert target.resolve() == original.resolve()
            elif target.exists():
                raise FileExistsError(f'Refusing to replace {target}')
            else:
                target.symlink_to(original)
            # Full decoding occurs once here, not at training startup or per epoch.
            with Image.open(original) as image:
                image.load()
                sizes.append(image.size)
                modes[image.mode] += 1
                if split != 'train':
                    pre_relative = Path(split + '_pre') / f'{cid:03d}' / (filename + '.png')
                    pre = root / pre_relative
                    pre.parent.mkdir(parents=True, exist_ok=True)
                    if not pre.exists():
                        resize(image.convert('RGB')).save(pre, format='PNG')
                    with Image.open(pre) as prepared:
                        prepared.load()
                        assert prepared.mode == 'RGB' and prepared.size == (84, 84)
                    record['preprocessed_relative_path'] = str(pre_relative)
            record.update(split=split, relative_path=str(relative),
                          imagefolder_label=class_to_idx[split][f'{cid:03d}'],
                          train_global_label=class_to_idx[split][f'{cid:03d}'] if split == 'train' else None)
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            counts[split] += 1
    partial.replace(manifest)
    assert [counts[s] for s in ('train', 'val', 'test')] == [10747, 1410, 4028]
    class_counts = Counter(r['original_class_id'] for r in records)
    assert min(class_counts.values()) >= 21
    for split in groups:
        ds = ImageFolder(root / split)
        assert ds.class_to_idx == class_to_idx[split] and len(ds) == counts[split]
        if split != 'train':
            ds_pre = ImageFolder(root / (split + '_pre'))
            assert ds_pre.class_to_idx == ds.class_to_idx and len(ds_pre) == len(ds)
    split_record = {'protocol_id': PROTOCOL, 'source_record': str(AUTHOR.parent / 'source.json'),
                    'original_ids': groups, 'class_to_idx': class_to_idx,
                    'class_names_by_id': {str(i+1): name for i, name in enumerate(names)},
                    'image_counts': dict(counts), 'per_class_images': dict(sorted(class_counts.items())),
                    'mapping': "annotation['class'] is original 1-based ID; meta.class_names[ID-1]; zero-padded directories sorted by ImageFolder; episode labels are independent 0..way-1"}
    write_json(prep / 'split.json', split_record)
    source_files = ['cars_annos.mat', 'devkit/cars_train_annos.mat',
                    'cars_test_annos_withlabels.mat', 'devkit/cars_meta.mat', 'devkit/README.txt']
    provenance = {'source_root': str(source), 'source_kind': 'existing local GitHub mirror copy',
                  'mirror': 'https://github.com/jhpohovey/StanfordCars',
                  'local_readme': str(source.parent / 'README.md'),
                  'original_archive_pixel_equivalence': 'not_verified',
                  'image_storage': 'symlinks to stable raw dataset; no bbox use, cropping or rewriting of originals',
                  'annotation_hashes': {p: digest(source / p) for p in source_files},
                  'source_readme_sha256': digest(source.parent / 'README.md'),
                  'original_image_modes': dict(modes), 'decoded_images': len(records),
                  'dimensions_min': [min(x[i] for x in sizes) for i in (0, 1)],
                  'dimensions_max': [max(x[i] for x in sizes) for i in (0, 1)],
                  'images_at_84x84': sum(x == (84, 84) for x in sizes)}
    write_json(prep / 'provenance.json', provenance)
    result = {'status': 'data_prepared', 'data_ready': True, 'protocol_id': PROTOCOL,
              'root': str(root), 'source_root': str(source), 'image_counts': dict(counts),
              'class_counts': {s: len(v) for s, v in groups.items()},
              'author_split_sha256': digest(AUTHOR), 'split_manifest': 'split.json',
              'sample_manifest': 'samples.jsonl', 'split_manifest_sha256': digest(prep / 'split.json'),
              'sample_manifest_sha256': digest(manifest), 'provenance_sha256': digest(prep / 'provenance.json'),
              'source_record_sha256': digest(AUTHOR.parent / 'source.json'),
              'full_decode_passed': True, 'seconds': time.perf_counter()-started,
              'preprocessing': 'RGB -> Resize(short edge=92) -> CenterCrop(84) -> PNG; loader only ToTensor/Normalize',
              'training_started': False}
    write_json(prepared_path, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    args = parser.parse_args()
    build(args.source.resolve())
