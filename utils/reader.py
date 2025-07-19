import mne
import numpy as np
import os

LABEL_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,  # merge with stage 3 -> N3
    'Sleep stage R': 4,
    'Movement time': -1,
    'Sleep stage ?': -1
}



def read_record(psg_path, hypnogram_path, channel='EEG Fpz-Cz'):
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations)

    if channel not in raw.ch_names:
        raise ValueError(f"Channel '{channel}' not found. Available: {raw.ch_names}")

    raw.pick_channels([channel])
    raw.resample(64)

    events, _ = mne.events_from_annotations(raw, event_id=LABEL_MAP)
    events = [e for e in events if e[2] != -1]
    return raw, events
