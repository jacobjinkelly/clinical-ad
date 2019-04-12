#!/bin/bash

#module load python/3.6.3
module load tensorflow/1.9.0-py3.5-cpu
cd /hpf/projects/brudno/marta/mimic_rs_collection/get_conceptEmbeds


python3 localglobalembed_forConceptEmbeds.py -id $PBS_ARRAYID -window 2 -all -g

