#!/bin/bash

module load python/3.6.3
cd /hpf/projects/brudno/marta/mimic_rs_collection


python3 link_cui_to_rs_dataset.py -id $PBS_ARRAYID

