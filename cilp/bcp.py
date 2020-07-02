import json
import re
import tempfile
import time
from os import path as osp

from .utils import aleph_settings, create_script, load_examples, run_aleph, pjoin


def run_bcp(data_dir, cached=True, print_output=False):
    bc_file = pjoin(data_dir, 'bc.json')
    if osp.exists(bc_file) and cached:
        print('Loading from cache')
        return

    bottom_clauses = {}
    for posneg in ['pos', 'neg']:
        bottom_clauses[posneg] = []

        train_pos = pjoin(data_dir, f'{posneg}.pl')
        pos_examples = load_examples(train_pos)
        bk_file = pjoin(data_dir, 'bk.pl')
        mode_file = pjoin(data_dir, 'mode.pl')

        script_lines = aleph_settings(mode_file, bk_file, data_files={'train_pos': train_pos})
        # script_lines += [f':- set(train_pos, "{train_pos}").']
        for i in range(len(pos_examples)):
            script_lines += [f':- sat({i+1}).']

        temp_dir = tempfile.mkdtemp()
        script_file = create_script(temp_dir, script_lines)

        print(f'Running Prolog script {script_file}')
        start_time = time.time()
        prolog_output = run_aleph(script_file)
        time_elapsed = time.time() - start_time
        print(f'Prolog done, took {time_elapsed:.1f} parsing output...')

        if print_output:
            print(prolog_output)

        bottom_clauses_raw = re.findall(r'\[bottom clause\]\n(.*?)\n\[literals\]', prolog_output,
                                        re.S)

        for b in bottom_clauses_raw:
            clause = re.sub(r'[ \n]', '', b).split(':-')
            if len(clause) == 1:
                continue
            body = clause[1]
            body = re.findall(r'(\w+\([\w,]+\))', body)
            bottom_clauses[posneg].append(sorted(body))

    with open(bc_file, 'w') as f:
        json.dump(bottom_clauses, f, indent=4)
