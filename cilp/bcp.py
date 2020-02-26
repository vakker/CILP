import json
import logging
import re
import tempfile
from os import path as osp

from .utils import (aleph_settings, create_script, execute, get_aleph,
                    load_examples, run_aleph)


def run_bcp(data_dir, cached=True, print_output=False):
    bc_file = osp.join(data_dir, 'bc.json')
    if osp.exists(bc_file) and cached:
        logging.info('Loading from cache')
        return

    bottom_clauses = {}
    for posneg in ['pos', 'neg']:
        bottom_clauses[posneg] = []

        train_pos = osp.join(data_dir, f'{posneg}.pl')
        pos_examples = load_examples(train_pos)
        bk_file = osp.join(data_dir, 'bk.pl')
        mode_file = osp.join(data_dir, 'mode.pl')

        script_lines = aleph_settings(mode_file,
                                      bk_file,
                                      data_files={'train_pos': train_pos})
        # script_lines += [f':- set(train_pos, "{train_pos}").']
        for i in range(len(pos_examples)):
            script_lines += [f':- sat({i+1}).']

        temp_dir = tempfile.mkdtemp()
        script_file = create_script(temp_dir, script_lines)

        logging.info('Running Prolog script...')
        prolog_output = run_aleph(script_file)
        logging.info('Prolog done, parsing output...')

        if print_output:
            logging.debug(prolog_output)

        bottom_clauses_raw = re.findall(
            r'\[bottom clause\]\n(.*?)\n\[literals\]', prolog_output, re.S)

        for b in bottom_clauses_raw:
            clause = re.sub(r'[ \n]', '', b).split(':-')
            if len(clause) == 1:
                continue
            body = clause[1]
            body = re.findall(r'(\w+\([\w,]+\))', body)
            bottom_clauses[posneg].append(sorted(body))

    with open(bc_file, 'w') as f:
        json.dump(bottom_clauses, f, indent=4)
