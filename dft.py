import prettier_console as pc
import soundfile as sf
import numpy as np
import tempfile
import os
import subprocess


frame_length_hz = 5
frame_frequency_hz = 60
lower_f = 20
upper_f = 20000
bar_count = 128

file_path = pc.select_files(file_types=["wav"])

data, sample_rate = sf.read(*file_path, dtype="float32")
if data.ndim == 2:
    data = data.mean(axis=1).astype("float32")

tmp = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
data.tofile(tmp)                 # raw float32 for the C++ side
fpath = tmp.name
tmp.close()

sample_per_frame   = int(sample_rate // frame_length_hz)
dframe = int(sample_rate // frame_frequency_hz)
freqs = np.geomspace(lower_f, upper_f, bar_count)           # same as your loop

here = os.path.dirname(os.path.abspath(__file__))
exe  = os.path.join(here, "executables", "dft.exe")
args = [exe, 
        fpath, 
        str(sample_rate), 
        str(sample_per_frame), 
        str(dframe),
        str(len(data)), 
        str(bar_count)] + \
        [f"{f:.6f}" for f in freqs]                         # one list item per argument

try:
    p = subprocess.run(args, capture_output=True)           # waits; stdout comes back as bytes
finally:
    os.remove(fpath)                                        # clean up the temp file

if p.returncode != 0:
    raise RuntimeError(p.stderr.decode(errors="replace"))

out = np.frombuffer(p.stdout, dtype=np.float32)             # flat: frames × bar_count values
out = out.reshape(-1, bar_count)                            # rows = frames, columns = frequencies