#!/bin/bash

module load python/3.6.3
cd /hpf/projects/brudno/marta/mimic_rs_collection

#PBS_ARRAYID=3

#for id in $(seq 1 $PBS_ARRAYID);
	#do
		#echo $id
		#python3 find_lines_umlsTargetword_mimic.py -id $id
	#done

python3 find_lines_umlsTargetword_mimic.py -id $PBS_ARRAYID

