import os
import re
import json
import csv
from typing import final

import numpy as np
from musicnn.extractor import extractor
from pyod.models.copod import COPOD
from pyod.utils.data import evaluate_print

music_extensions = ['mp3', 'wav', 'flac', 'm4a']
music_ext_pattern = '^' + '|'.join([
    '(.*\.' + ''.join([
        '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
        for c in ext])
    + ')' + '$'
    for ext in music_extensions])


def tags_vec(file_path):
    taggram, tags, _ = extractor(
        file_path, model='MSD_musicnn_big')
    return tags, np.mean(taggram, axis=0).tolist()


def init_data(tags=None):
    return {
        'tags': tags.copy() if tags else None,
        'path': [],
        'vecs': [],
    }


def index_dir(dir):
    data = init_data()
    for root, _, files in os.walk(dir):
        for filename in files:
            if not re.match(music_ext_pattern, filename):
                continue
            file_path = os.path.join(root, filename)
            tags, vec = tags_vec(file_path)
            if data['tags'] is None:
                data['tags'] = tags
            data['path'].append(file_path)
            data['vecs'].append(vec)
    return data


def dump_data(data_file, data):
    with open(data_file, 'w') as output:
        output.write(json.dumps(data))


def load_data(data_file):
    with open(data_file, 'r') as o:
        return json.load(o)


def album_subdata(lib_data, lib_dir, album_names):
    data = init_data(lib_data['tags'])
    for name in album_names:
        album_path = os.path.join(lib_dir, name)
        for subpath, _, files in os.walk(album_path):
            for filename in files:
                file_path = os.path.join(subpath, filename)
                file_index = lib_data['path'].index(file_path)
                if file_index >= 0:
                    data['path'].append(file_path)
                    data['vecs'].append(lib_data['vec'][file_index].copy())
    return data


def album_subdata(lib_data, lib_dir, album_names):
    data = init_data(lib_data['tags'])
    for name in album_names:
        album_path = os.path.join(lib_dir, name)
        for subpath, _, files in os.walk(album_path):
            for filename in files:
                if not re.match(music_ext_pattern, filename):
                    continue
                file_path = os.path.join(subpath, filename)
                try:
                    file_index = lib_data['path'].index(file_path)
                except Exception:
                    pass
                else:
                    data['path'].append(file_path)
                    data['vecs'].append(lib_data['vecs'][file_index].copy())
    return data


def album_pickdata(pick_to, lib_data, lib_dir, album_names):
    for name in album_names:
        album_path = os.path.join(lib_dir, name)
        for subpath, _, files in os.walk(album_path):
            for filename in files:
                if not re.match(music_ext_pattern, filename):
                    continue
                file_path = os.path.join(subpath, filename)
                try:
                    file_index = lib_data['path'].index(file_path)
                except Exception:
                    pass
                else:
                    pick_to['path'].append(file_path)
                    pick_to['vecs'].append(lib_data['vecs'][file_index].copy())
                    break
    return pick_to


def album_od(lib_data, lib_dir, inlier_albums, outlier_albums=[], output='out.csv', weighted=True):
    data = album_subdata(lib_data, lib_dir, inlier_albums)
    inlier_count = len(data['vecs'])
    album_pickdata(data, lib_data, lib_dir, outlier_albums)
    n = len(data['vecs'])
    outlier_count = n - inlier_count
    label_truth = np.r_[np.zeros(inlier_count), np.ones(outlier_count)]

    data_mat = np.array(data['vecs'])

    clf = COPOD(contamination=outlier_count /
                n) if outlier_count > 1 else COPOD()
    clf.fit(data_mat)

    od_scores = clf.decision_scores_
    od_mat = clf.O[-n:]
    median_likelihood = np.median(data_mat, axis=0)
    if weighted:
        weights = np.maximum(data_mat, median_likelihood.T)
        od_mat = np.multiply(od_mat, weights)
        od_scores = od_mat.sum(axis=1)

    if len(outlier_albums) > 0:
        evaluate_print('Genre COPOD', label_truth, clf.labels_)

    with open(output, 'w+') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow([
            'Truth',
            'Fitted',
            'OD',
            'File Path',
            'Judgement: Dimentional OD Score (Likelihood Score / Median Likelihood Score)'
        ])
        csvWriter.writerows([[
            "%d" % label_truth[i],
            "%d" % clf.labels_[i],
            "%.3lf" % od_scores[i],
            os.path.relpath(data['path'][i], lib_dir),
            ", ".join([
                "%s %s:%.3lf(%.3lf/%.3lf)" % pair
                for pair in sorted([
                    (
                        'more' if data['vecs'][i][j] > median_likelihood[j] else 'less',
                        tag,
                        od_mat[i][j],
                        data['vecs'][i][j],
                        median_likelihood[j]
                    )
                    for j, tag in enumerate(data['tags'])
                ], key=lambda x: x[2], reverse=True)
            ])] for i in range(n)])


if __name__ == '__main__':
    itunes_dir = '/mnt/d/Music/iTunes/iTunes Media/Music/'

    # data = index_dir(itunes_dir)
    # dump_data('itunes.json', data)

    lib_data = load_data('itunes.json')

    album_od(lib_data, itunes_dir, ['the distant journey to you'], [
        'Foxtail-Grass Studio', 'TUMENECO', 'Halozy', 'DJ OKAWARI', 'Nami Haven'])
    # album_od(lib_data, itunes_dir, ['Nami Haven'])
