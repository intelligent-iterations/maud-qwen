#!/usr/bin/env python3
"""Independent temperature interlock. Pause preserves the training process and state."""
import argparse
import datetime
import http.client
import json
from pathlib import Path
import signal
import socket
import subprocess
import time


class DockerConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect('/var/run/docker.sock')


def docker(method, path):
    connection = DockerConnection('localhost', timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        body = response.read()
        if response.status >= 300:
            raise RuntimeError(f'Docker {response.status}: {body.decode()}')
        return json.loads(body) if body else None
    finally:
        connection.close()


def decision(reading, config):
    if reading['gpu_limit_w'] > config['expected_gpu_limit_w'] or reading['cpu_max_khz'] > config['expected_cpu_max_khz']:
        return 'limits_changed'
    if reading['cpu_c'] >= config['cpu_pause_c'] or reading['gpu_c'] >= config['gpu_pause_c'] or reading['gpu_thermal_slowdown']:
        return 'hot'
    if reading['cpu_c'] <= config['cpu_resume_c'] and reading['gpu_c'] <= config['gpu_resume_c']:
        return 'cool'
    return 'hold'


def sensors(sysroot):
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,power.limit,fan.speed,clocks_event_reasons.sw_thermal_slowdown',
                             '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=5)
    if result.returncode:
        raise RuntimeError(f'GPU telemetry exit {result.returncode}: {result.stdout.strip()} {result.stderr.strip()}')
    rows = result.stdout.strip().splitlines()
    assert len(rows) == 1
    temp, watts, limit, fan, thermal = [v.strip() for v in rows[0].split(',')]
    cpu = []
    for hw in (sysroot/'class/hwmon').glob('hwmon*'):
        if (hw/'name').read_text().strip() == 'k10temp':
            cpu.extend(float(v.read_text())/1000 for v in hw.glob('temp*_input'))
    assert cpu, 'CPU temperature sensor missing'
    caps = [int(v.read_text()) for v in (sysroot/'devices/system/cpu/cpufreq').glob('policy*/scaling_max_freq')]
    assert caps, 'CPU frequency policy missing'
    reading = {'cpu_c': max(cpu), 'gpu_c': float(temp), 'gpu_watts': float(watts), 'gpu_limit_w': float(limit),
               'gpu_fan_percent': float(fan), 'gpu_thermal_slowdown': thermal == 'Active', 'cpu_max_khz': max(caps)}
    assert 0 < reading['cpu_c'] < 120 and 0 < reading['gpu_c'] < 120
    assert thermal in ('Active', 'Not Active')
    return reading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--sys-root', type=Path, default=Path('/host-sys'))
    parser.add_argument('--project-dir', type=Path)
    parser.add_argument('--resume-initial-pause', action='store_true')
    a = parser.parse_args()
    config = json.loads(a.config.read_text())
    assert config['cpu_resume_c'] < config['cpu_pause_c'] and config['gpu_resume_c'] < config['gpu_pause_c']
    assert config['sample_seconds'] > 0 and config['cool_samples_to_resume'] > 0
    endpoint = '/containers/'+a.container
    a.evidence.mkdir(parents=True, exist_ok=True)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    owns_pause = a.resume_initial_pause
    state_path = a.evidence/'guard_state.json'
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        assert previous['container'] == a.container
        owns_pause = previous.get('owns_pause', False)
    paused_at = time.monotonic()
    cool_samples = 0
    def record(event, **details):
        value = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'event': event,
                 'container': a.container, 'owns_pause': owns_pause, **details}
        if a.project_dir and event in ('paused', 'resumed'):
            progress = {}
            for name in ['outputs/untuned-qwen/validation.jsonl', 'outputs/initial/steps.jsonl',
                         'outputs/untuned-qwen/test.jsonl', 'outputs/tuned-qwen/test.jsonl']:
                path = a.project_dir/name
                if path.exists():
                    with path.open() as source:
                        lines = source.readlines()
                    progress[name] = {'completed_lines': len(lines)}
                    if name.endswith('steps.jsonl') and lines:
                        progress[name]['last_completed_global_step'] = json.loads(lines[-1])['global_step']
            value['progress'] = progress
        with (a.evidence/'thermal_events.jsonl').open('a') as out:
            out.write(json.dumps(value)+'\n'); out.flush()
        tmp = state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(state_path)
        if event != 'sample': print(json.dumps(value), flush=True)
    def pause(reason):
        nonlocal owns_pause, paused_at, cool_samples
        state = docker('GET', endpoint+'/json')['State']
        if state['Running'] and not state['Paused']:
            docker('POST', endpoint+'/pause')
            owns_pause = True
            paused_at = time.monotonic()
            cool_samples = 0
            record('paused', reason=reason)
    record('started', config=config)
    try:
        while not stopping:
            state = docker('GET', endpoint+'/json')['State']
            if not state['Running']:
                record('target_exited', exit_code=state['ExitCode'])
                return
            try:
                reading = sensors(a.sys_root)
                action = decision(reading, config)
            except Exception as error:
                pause('sensor_failure')
                cool_samples = 0
                record('sensor_failure', error=str(error))
                time.sleep(config['sample_seconds'])
                continue
            if action in ('hot', 'limits_changed'):
                pause(action)
                cool_samples = 0
            elif state['Paused'] and owns_pause:
                cool_samples = cool_samples+1 if action == 'cool' else 0
                if cool_samples >= config['cool_samples_to_resume'] and time.monotonic()-paused_at >= config['minimum_pause_seconds']:
                    docker('POST', endpoint+'/unpause')
                    owns_pause = False
                    record('resumed', reading=reading)
            record('sample', reading=reading, decision=action, paused=docker('GET', endpoint+'/json')['State']['Paused'])
            time.sleep(config['sample_seconds'])
    finally:
        # A stopped or failed guard leaves this workload paused, never unmonitored.
        pause('guard_stopping')
        record('stopped')


if __name__ == '__main__':
    main()
