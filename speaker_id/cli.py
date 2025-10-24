from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import sys
import numpy as np

from .audio import load_audio_mono
from .features import compute_mfcc, FeatureConfig
from .storage import Storage
from .model import SimpleSpeakerId


def cmd_enroll(args: argparse.Namespace) -> int:
    storage = Storage(Path(args.db))
    storage.ensure()

    audio, sr = load_audio_mono(args.audio, target_sr=args.sample_rate)
    feat = compute_mfcc(audio, FeatureConfig(sample_rate=args.sample_rate))

    storage.add(args.name, feat)
    print(f"Enrolled '{args.name}' from {args.audio}.")
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    storage = Storage(Path(args.db))
    enrollments = storage.all_enrollments()
    if not enrollments:
        print("No enrolled speakers. Use 'enroll' first.")
        return 2

    audio, sr = load_audio_mono(args.audio, target_sr=args.sample_rate)
    feat = compute_mfcc(audio, FeatureConfig(sample_rate=args.sample_rate))

    model = SimpleSpeakerId(threshold=args.threshold)
    name, score, scores = model.identify(feat, enrollments)

    if name is None:
        print(f"Unknown speaker (best score={score:.3f}).")
        if args.verbose:
            for n, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {n}: {s:.3f}")
        return 1

    print(f"Predicted: {name} (score={score:.3f})")
    if args.verbose:
        for n, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {n}: {s:.3f}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    storage = Storage(Path(args.db))
    names = storage.list_speakers()
    if not names:
        print("No enrolled speakers.")
        return 0
    for n in names:
        print(n)
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    storage = Storage(Path(args.db))
    ok = storage.remove(args.name)
    if ok:
        print(f"Removed '{args.name}'.")
        return 0
    else:
        print(f"Speaker '{args.name}' not found.")
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="speaker-id", description="Simple speaker identification (MFCC + cosine)")
    p.add_argument("--db", type=str, default=str(Path.home() / ".speaker_id_db"), help="Path to storage directory")
    p.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")

    sub = p.add_subparsers(dest="cmd", required=True)

    # enroll
    pe = sub.add_parser("enroll", help="Enroll a speaker from an audio file")
    pe.add_argument("name", type=str, help="Speaker name")
    pe.add_argument("audio", type=str, help="Path to audio file (wav/mp3/etc)")
    pe.set_defaults(func=cmd_enroll)

    # identify
    pi = sub.add_parser("identify", help="Identify speaker of an audio file")
    pi.add_argument("audio", type=str, help="Path to audio file (wav/mp3/etc)")
    pi.add_argument("--threshold", type=float, default=0.55, help="Cosine similarity threshold for unknown")
    pi.add_argument("-v", "--verbose", action="store_true")
    pi.set_defaults(func=cmd_identify)

    # list
    pl = sub.add_parser("list", help="List enrolled speakers")
    pl.set_defaults(func=cmd_list)

    # remove
    pr = sub.add_parser("remove", help="Remove a speaker by name")
    pr.add_argument("name", type=str)
    pr.set_defaults(func=cmd_remove)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
