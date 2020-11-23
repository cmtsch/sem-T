#!/bin/bash

#ExperimentName='NewTrainingFile'
#ModelTag='BNL_DV'
ExperimentName=$1
ModelTag=$2
#echo $ExperimentName
#echo $ModelTag

checkpointNr=0
while [  -f data/models/${ExperimentName}/${ExperimentName}_${ModelTag}/checkpoint_${checkpointNr}.pth ]; do
  #echo "../data/models/${ExperimentName}/${ExperimentName}_${ModelTag}/checkpoint_${checkpointNr}.pth"
  let checkpointNr=$checkpointNr+1
  #echo $checkpointNr
done

if [ ${checkpointNr} == 0 ]; then
  echo "Model probably does not exist"
  exit
fi

let checkpointNr=$checkpointNr-1
#echo $checkpointNr

if [[ $ModelTag =~ "H" ]]; then
  Dimensions='High'
else
  Dimensions='Low'
fi


bsub -R "rusage[ngpus_excl_p=4, mem=64000]" -oo Hpatches_Desc${ExperimentName}_${ModelTag} \
python benchmarks/hpatches_extract_HardNet.py /cluster/scratch/cmitsch/hpatches-release \
/cluster/scratch/cmitsch/csvDesc/ data/models/${ExperimentName}/${ExperimentName}_${ModelTag}/checkpoint_${checkpointNr}.pth \
${ExperimentName}_${ModelTag} ${Dimensions}