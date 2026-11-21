import prettier_console as pc
import soundfile as sf
import numpy as np
import tempfile
from sys import stderr, stdin, stdout

frame_length_hz = 10
f1 = 20
f2 = 20000
N = 32
freqs = []

file_path = pc.select_files(file_types=["wav"])

data, fs = sf.read(*file_path, dtype="float32")
if data.ndim == 2:
    data = data.mean(axis=1).astype("float32")

tmp = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
data.tofile(tmp)                 # raw float32 for the C++ side
path = tmp.name
tmp.close()

for i in range(0, N):
    freqs.append(f1 * (f2 / f1) ** (i / (N - 1)))

config = "\n".join(f"{path}", 
                   f"{fs} {len(data)} {fs/frame_length_hz} {len(freqs)}" +
                   " ".join(f"{f:.6f}" for f in freqs), "")
stdout.write(config)
stdin.read()

