import json
from os import path

self_dir = path.dirname(path.realpath(__file__))
# load indices of residues
with open(path.join(self_dir, 'residues.json'), 'r') as fin:
    res_dict = json.load(fin)
