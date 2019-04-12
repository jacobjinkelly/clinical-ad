#!/bin/bash

module load python/3.6.3
cd /hpf/projects/brudno/marta/mimic_rs_collection


python3 delete_repeated_samples.py -id $PBS_ARRAYID

