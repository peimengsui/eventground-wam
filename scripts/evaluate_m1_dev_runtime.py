#!/usr/bin/env python3
"""Use the production-qualified desktop GPU check with the frozen M1 evaluator."""
from __future__ import annotations

import os
import subprocess

import evaluate_m1_dev as evaluation
from train_auxiliary_policy import other_gpu_jobs


def require_idle_gpu():
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
         '--format=csv,noheader,nounits'], text=True)
    blocked = other_gpu_jobs(processes, os.getpid())
    evaluation.require(not blocked, f'another substantial GPU job is running: {blocked}')


def main():
    evaluation.require_idle_gpu = require_idle_gpu
    evaluation.main()


if __name__ == '__main__':
    main()
