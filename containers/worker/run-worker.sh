#!/bin/bash

/worker/pull-model-data.sh
/usr/bin/python3 /worker/poem-worker.py
