#!/bin/bash

module load python/3.6.3
cd /hpf/projects/brudno/marta/mimic_rs_collection


python3 sample_cui_terms.py -id $PBS_ARRAYID

