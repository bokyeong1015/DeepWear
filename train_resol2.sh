#!/bin/bash

th train.lua -expNum 1 -resol 2 -vgg 1 \
-continueFlag 0 -nEpoch 1500 -learningRate 0.0002 \
-batchSize 10 -batchSize_eval 10

th train.lua -expNum 1 -resol 2 -vgg 1 \
-continueFlag 1 -nEpoch 5000 -learningRate 0.0001 \
-batchSize 10 -batchSize_eval 10

th train.lua -expNum 1 -resol 2 -vgg 1 \
-continueFlag 1 -nEpoch 6000 -learningRate 0.00005 \
-batchSize 10 -batchSize_eval 10

th train.lua -expNum 1 -resol 2 -vgg 1 \
-continueFlag 1 -nEpoch 7000 -learningRate 0.000025 \
-batchSize 10 -batchSize_eval 10