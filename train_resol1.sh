#!/bin/bash

th train.lua -expNum 1 -resol 1 -vgg 1 \
-continueFlag 0 -nEpoch 2000 -learningRate 0.0002 \
-batchSize 30 -batchSize_eval 30

th train.lua -expNum 1 -resol 1 -vgg 1 \
-continueFlag 1 -nEpoch 4000 -learningRate 0.0001 \
-batchSize 30 -batchSize_eval 30

th train.lua -expNum 1 -resol 1 -vgg 1 \
-continueFlag 1 -nEpoch 5000 -learningRate 0.00005 \
-batchSize 30 -batchSize_eval 30

th train.lua -expNum 1 -resol 1 -vgg 1 \
-continueFlag 1 -nEpoch 6000 -learningRate 0.000025 \
-batchSize 30 -batchSize_eval 30