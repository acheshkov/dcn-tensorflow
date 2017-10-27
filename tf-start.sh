#!/bin/bash

jupyter notebook &
tensorboard --logdir=/tmp/working/logs --host=0.0.0.0 --port=8080 &
