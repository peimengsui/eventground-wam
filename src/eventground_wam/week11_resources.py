"""Shared qualification ledger and hard process-tree resource guard."""
from __future__ import annotations

import csv
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import torch

from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.clean_comparison import atomic_json
from eventground_wam.demonstration_contracts import seal_contract


class QualificationStopped(BaseException):
    """Must escape policy/infrastructure Exception handlers without a retry."""


def competing_gpu_jobs(processes, own_pid):
    """Feifei lists idle desktop graphics clients as compute processes."""
    blocked = []
    for row in csv.reader(processes.splitlines()):
        pid, name, memory = (value.strip() for value in row)
        if int(pid) == own_pid:
            continue
        desktop = (name in ('/usr/bin/gnome-control-center', '/usr/bin/ptyxis')
                   or name.startswith('/opt/google/chrome/chrome --type=gpu-process '))
        if not (desktop and memory.isdigit() and int(memory) <= 256):
            blocked.append(int(pid))
    return blocked


def require_idle_gpu():
    processes = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    blocked = competing_gpu_jobs(processes, os.getpid())
    if blocked:
        raise ValueError(f'Another GPU compute job is running: {blocked}')
    return processes


class ResourceGuard:
    def __init__(self, output, identity, limits):
        self.output, self.limits = Path(output), limits
        self.output.mkdir(parents=True, exist_ok=True)
        self.budget = ActiveBudget(self.output/'active-budget.json', identity)
        self.peak_rss = self.peak_allocated = self.peak_reserved = 0
        self.output_bytes = 0
        self.free_bytes = shutil.disk_usage(self.output).free
        self.reason = None
        self.stop = threading.Event()
        self.previous = signal.signal(signal.SIGUSR1, self.interrupt)
        self.previous_term = signal.signal(signal.SIGTERM, self.interrupt)
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def interrupt(self, *_):
        raise QualificationStopped(self.reason or 'External interruption; preserve attempt')

    def snapshot(self):
        return {'active_seconds': self.budget.seconds, 'peak_process_tree_rss_bytes': self.peak_rss,
                    'peak_allocated_bytes': self.peak_allocated, 'peak_reserved_bytes': self.peak_reserved,
                    'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_initialized() else None,
                    'output_bytes': self.output_bytes, 'free_bytes': self.free_bytes, 'stop_reason': self.reason}

    def check(self):
        if self.reason:
            raise QualificationStopped(self.reason)
        if self.budget.seconds >= self.limits['qualification_active_seconds']:
            raise QualificationStopped('Aggregate qualification active-time ceiling')

    def monitor(self):
        next_disk = 0.
        try:
            import psutil

            while not self.stop.wait(.2):
                process = psutil.Process()
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                self.peak_rss = max(self.peak_rss, rss)
                if rss > self.limits['host_process_tree_rss_limit_bytes']:
                    raise RuntimeError('Process-tree RSS ceiling')
                if torch.cuda.is_initialized():
                    self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
                    self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())
                    total = torch.cuda.get_device_properties(0).total_memory
                    if max(self.peak_allocated, self.peak_reserved) >= total*self.limits['gpu_peak_fraction_strictly_below']:
                        raise RuntimeError('GPU allocation/reservation ceiling')
                if time.monotonic() >= next_disk:
                    require_idle_gpu()
                    self.output_bytes = 0
                    for path in self.output.rglob('*'):
                        try:
                            if path.is_file():
                                self.output_bytes += path.stat().st_size
                        except FileNotFoundError:  # atomic recorder/ledger rename in flight
                            pass
                    self.free_bytes = shutil.disk_usage(self.output).free
                    if self.output_bytes >= self.limits['output_limit_bytes']:
                        raise RuntimeError('Output storage ceiling')
                    if self.free_bytes < self.limits['minimum_free_bytes']:
                        raise RuntimeError('Free-disk reserve ceiling')
                    atomic_json(self.output/'resources-live.json', seal_contract(self.snapshot()))
                    next_disk = time.monotonic()+5
                self.check()
        except BaseException as exc:  # noqa: BLE001 - watchdog failures must stop the run
            self.reason = f'{type(exc).__name__}: {exc}'
            os.kill(os.getpid(), signal.SIGUSR1)

    def close(self):
        self.stop.set()
        self.thread.join()
        snapshot = self.snapshot()
        self.budget.close()
        signal.signal(signal.SIGUSR1, self.previous)
        signal.signal(signal.SIGTERM, self.previous_term)
        return snapshot
