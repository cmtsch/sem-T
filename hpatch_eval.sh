#!/bin/bash
ExperimentName=$1
ModelTag=$2
helpTask=$3
if [[ ${helpTask} == "vonMises" ]]; then
  task="--task=vonMises"
else
  task="--task=verification --task=matching --task=retrieval --task=vonMises"
fi

descrName=${ExperimentName}_${ModelTag}

bsub -n 1 -R "rusage[mem=8400]" -oo HpatchesEval_${ExperimentName}_${ModelTag} \
python ../hpatches-benchmark/python/hpatches_eval.py --descr-name=${descrName} \
--descr-dir=/cluster/scratch/cmitsch/csvDesc/ \
--results-dir=../hpatches-benchmark/python/results/ ${task}