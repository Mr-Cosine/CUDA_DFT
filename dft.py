# TODO: add super resolution to the post processing.

import prettier_console as pc
import soundfile as sf
import sounddevice as sd
import numpy as np
import ctypes
import os
import time
import json

frame_length_hz = 5
frame_frequency_hz = 30
lower_f = 20
upper_f = 20000
bar_count = 16

file_path = pc.select_files(file_types=["wav"])

data, sample_rate = sf.read(*file_path, dtype="float32")
if data.ndim == 2:
    data = data.mean(axis=1).astype("float32")

data = np.ascontiguousarray(data, dtype=np.float32)         # C side reads a flat float32 buffer

sample_per_frame   = int(sample_rate // frame_length_hz)
dframe = int(sample_rate // frame_frequency_hz)
freqs = np.geomspace(lower_f, upper_f, bar_count).astype(np.float32)

# ---- load the CUDA DLL ----
lib  = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "dft.dll"))

c_float_p = ctypes.POINTER(ctypes.c_float)
lib.audio_dft.argtypes = [c_float_p,                        # output (frames × bar_count)
                          c_float_p,                        # samples
                          ctypes.c_int,                     # sample_rate
                          ctypes.c_int,                     # sample_per_frame
                          ctypes.c_int,                     # distance_frames
                          ctypes.c_int,                     # total_sample_count
                          ctypes.c_int,                     # bar_count
                          c_float_p]                        # frequencies
lib.audio_dft.restype  = ctypes.c_int
lib.error_recall.argtypes = []
lib.error_recall.restype  = ctypes.c_char_p

frame_count = (len(data) + dframe - 1) // dframe            # must match main.cu
out = np.empty((frame_count, bar_count), dtype=np.float32)  # rows = frames, columns = frequencies

rc = lib.audio_dft(out.ctypes.data_as(c_float_p),
                   data.ctypes.data_as(c_float_p),
                   int(sample_rate),
                   sample_per_frame,
                   dframe,
                   len(data),
                   bar_count,
                   freqs.ctypes.data_as(c_float_p))
if rc != 0:
    raise RuntimeError(lib.error_recall().decode(errors="replace"))

db = 20 * np.log10(out + 1e-9)                              # amplitude → dB (1e-9 avoids log(0))

DB_MIN, DB_MAX = -80, 0                                     # display range, tune by eye
norm_dB = np.clip((db - DB_MIN) / (DB_MAX - DB_MIN), 0, 1)  # 0..1 bar heights

song = {
        "info":
            {
                "wav_path": "",
                "title": "Track",
                "name": (os.path.split(*file_path)[-1]).replace(".wav", ""), 
                "artist": "Artist",
                "frame_frequency": frame_frequency_hz
            },
        "data": []
        }
for i, row in enumerate(norm_dB): 
    frame = []
    for bin in row:
        frame.append(float(bin))
    song["data"].append(frame)

folder_path = pc.select_folder(prompt="select where you want to export the processed file to")

with open(os.path.join(folder_path, f"{song["info"]["name"]}.json"), "w") as file:
    json.dump(song, file, indent=4)






pc.clear_screen()
max_bar_length = 20
def printbar(row):
    frame = ""
    for i,b in enumerate(row):
        frequency = f"{freqs[i]:.3f}"
        frame += frequency + " "*(10-len(frequency)) + ": " + "█" * int(b * max_bar_length) + "\n"
    pc.clear_lines(len(row) + 1)
    print(frame, flush=True)

period = 1 / frame_frequency_hz
next_t = time.perf_counter()

sd.play(data, sample_rate)
t0 = time.perf_counter()
while (f := int((time.perf_counter() - t0) * frame_frequency_hz)) < len(out):
    printbar(norm_dB[f])
    time.sleep(max(0.0, t0 + (f + 1) / frame_frequency_hz - time.perf_counter()))
