import subprocess
import time



filename = "tmp.sh"
N = 100

with open(filename) as f:
    commands = [
        line.strip()
        for line in f
        if line.strip() and not line.strip().startswith("#")
    ]

print(commands)

running = []

for cmd in commands:
    while len(running) >= N:
        running = [p for p in running if p.poll() is None]
        # time.sleep(0.5)
        time.sleep(2.0)

    print("start:", cmd)
    running.append(subprocess.Popen(cmd, shell=True))

for p in running:
    p.wait()