#!/bin/bash

for i in {0..15}
do
   cat jointTrainingSetup.py > batchjointlytrainer.py
   echo "i = $i" >> batchjointlytrainer.py
   cat trainjointmodelsnippet.py >> batchjointlytrainer.py
   ipython batchjointlytrainer.py
done
