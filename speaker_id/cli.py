from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Literal
import sys
import numpy as np

from .audio import load_audio_mono
from .features import FeatureConfig
from .storage import Storage
from .model import SimpleSpeakerId, compute_znorm
from .preprocess import preprocess_audio, PreprocessConfig
from .embeddings import EmbeddingExtractor, EmbeddingConfig
from .diarize import diarize_embeddings


BackendChoice = Literal["auto", "mfcc", "ecapa"]


def select_backend(choice: BackendChoice) -> str:
    if choice == "auto":
        try:
            import speechbrain  # type: ignore
            return "ecapa"
        except Exception:
            return "mfcc"
    return choice


def build_extractor(args: argparse.Namespace) -> EmbeddingExtractor:
    backend = select_backend(args.backend)
    return EmbeddingExtractor(EmbeddingConfig(sample_rate=args.sample_rate, backend=backend))


def get_preproc_config(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        sample_rate=args.sample_rate,
        trim_top_db=args.trim_top_db,
        pre_emphasis=args.pre_emphasis,
        target_rms=args.target_rms,
        noise_reduction=args.noise_reduction,
        nr_n_fft=args.nr_n_fft,
        nr_hop_length=args.nr_hop_length,
        nr_percentile=args.nr_percentile,
        use_webrtc_vad=args.vad,
        vad_aggressiveness=args.vad_aggr,
        vad_frame_ms=args.vad_frame_ms,
    )


def maybe_preprocess(audio: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.no_preprocess:
        return audio
    return preprocess_audio(audio, get_preproc_config(args))


def cmd_enroll(args: argparse.Namespace) -> int:
    storage = Storage(Path(args.db))
    storage.ensure()

    audio, sr = load_audio_mono(args.audio, target_sr=args.sample_rate)
    audio = maybe_preprocess(audio, args)
    extractor = build_extractor(args)
    feat = extractor.extract(audio)

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
    audio = maybe_preprocess(audio, args)
    extractor = build_extractor(args)
    feat = extractor.extract(audio)

    model = SimpleSpeakerId(threshold=args.threshold)
    name, score, scores = model.identify(feat, enrollments)
    if getattr(args, "znorm", False):
        templates = {n: model.enroll_vector(v) for n, v in enrollments.items()}
        impostor_means: dict[str, float] = {}
        impostor_stds: dict[str, float] = {}
        for n, t in templates.items():
            others = [templates[k] for k in templates.keys() if k != n]
            if not others:
                continue
            sims = []
            for o in others:
                denom = (np.linalg.norm(t) * np.linalg.norm(o)) + 1e-8
                sims.append(float(np.dot(t, o) / denom))
            impostor_means[n] = float(np.mean(sims))
            impostor_stds[n] = float(np.std(sims) + 1e-6)
        scores = compute_znorm(scores, impostor_means, impostor_stds)
        if name is not None:
            score = scores.get(name, score)

    if name is None:
        print(f"Unknown speaker (best score={score:.3f}).")
        if args.verbose or args.top_k > 0:
            for n, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k or None]:
                print(f"  {n}: {s:.3f}")
        return 1

    print(f"Predicted: {name} (score={score:.3f})")
    if args.verbose or args.top_k > 0:
        for n, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k or None]:
            print(f"  {n}: {s:.3f}")
    return 0


def cmd_mic(args: argparse.Namespace) -> int:
    try:
        import sounddevice as sd  # type: ignore
    except Exception:
        print("sounddevice not installed. Install mic extras or use requirements-extra.txt.")
        return 2

    duration = float(args.seconds)
    sr = args.sample_rate
    print(f"Recording {duration:.1f}s from microphone at {sr} Hz...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.reshape(-1)
    audio = maybe_preprocess(audio, args)
    extractor = build_extractor(args)
    feat = extractor.extract(audio)

    storage = Storage(Path(args.db))
    enrollments = storage.all_enrollments()
    if not enrollments:
        print("No enrolled speakers. Use 'enroll' first.")
        return 2
    model = SimpleSpeakerId(threshold=args.threshold)
    name, score, scores = model.identify(feat, enrollments)
    if name is None:
        print(f"Unknown speaker (best score={score:.3f}).")
        return 1
    print(f"Predicted: {name} (score={score:.3f})")
    return 0


def cmd_mic_live(args: argparse.Namespace) -> int:
    try:
        import sounddevice as sd  # type: ignore
    except Exception:
        print("sounddevice not installed. Install mic extras.")
        return 2
    sr = args.sample_rate
    storage = Storage(Path(args.db))
    enrollments = storage.all_enrollments()
    if not enrollments:
        print("No enrolled speakers. Use 'enroll' first.")
        return 2
    extractor = build_extractor(args)
    model = SimpleSpeakerId(threshold=args.threshold)

    block = int(args.window * sr)
    hop = int(args.hop * sr)
    if hop <= 0:
        hop = block
    buf = np.zeros(block, dtype=np.float32)
    idx = 0

    print(f"Listening live (window={args.window}s, hop={args.hop}s). Ctrl+C to stop.")

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32') as stream:
            while True:
                audio, _ = stream.read(hop)
                audio = audio.reshape(-1)
                # roll buffer
                n = min(hop, block)
                buf[:-n] = buf[n:]
                buf[-n:] = audio[:n]
                proc = maybe_preprocess(buf, args)
                feat = extractor.extract(proc)
                name, score, scores = model.identify(feat, enrollments)
                if name is None:
                    print(f"Unknown (score={score:.3f})")
                else:
                    print(f"{name} ({score:.3f})")
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


def cmd_diarize(args: argparse.Namespace) -> int:
    audio, sr = load_audio_mono(args.audio, target_sr=args.sample_rate)
    audio = maybe_preprocess(audio, args)
    extractor = build_extractor(args)

    # Build sliding windows
    win = int(1.5 * sr)
    hop = int(0.5 * sr)
    spans = list(range(0, max(1, len(audio) - win + 1), hop))
    if not spans:
        spans = [0]
    embs = []
    valid_spans = []
    for s in spans:
        e = min(len(audio), s + win)
        seg = audio[s:e]
        if len(seg) < int(0.3 * sr):
            continue
        embs.append(extractor.extract(seg))
        valid_spans.append((s, e))
    if not embs:
        print("Audio too short for diarization.")
        return 2
    embs = np.stack(embs, axis=0)
    labels = diarize_embeddings(embs, num_speakers=args.num_speakers)

    label_to_name = {}
    if args.assign:
        storage = Storage(Path(args.db))
        enrollments = storage.all_enrollments()
        # Build per-speaker templates using identification model aggregation
        model = SimpleSpeakerId()
        templates = {name: model.enroll_vector(vecs) for name, vecs in enrollments.items()}
        # Assign each cluster centroid to nearest template
        # Compute cluster centroids
        import collections
        cluster_to_indices = collections.defaultdict(list)
        for idx, lab in enumerate(labels):
            cluster_to_indices[int(lab)].append(idx)
        cluster_centroids = {
            lab: embs[idxs].mean(axis=0) if len(idxs) > 1 else embs[idxs[0]]
            for lab, idxs in cluster_to_indices.items()
        }
        # cosine scores
        from .model import cosine_similarity
        for lab, centroid in cluster_centroids.items():
            best = None
            best_s = -1.0
            for name, templ in templates.items():
                s = cosine_similarity(centroid, templ)
                if s > best_s:
                    best_s = s
                    best = name
            label_to_name[int(lab)] = best if best is not None else f"speaker_{lab}"

    for i, (s, e) in enumerate(valid_spans[: len(labels)]):
        start_s = s / sr
        end_s = e / sr
        lab = int(labels[i])
        disp = label_to_name.get(lab, f"speaker_{lab}")
        print(f"segment {i}: {start_s:.2f}-{end_s:.2f}s -> {disp}")
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
    p.add_argument("--backend", type=str, choices=["auto", "mfcc", "ecapa"], default="auto", help="Embedding backend")
    p.add_argument("--no-preprocess", action="store_true", help="Disable preprocessing (silence trim, pre-emphasis, RMS norm)")
    p.add_argument("--trim-top-db", type=float, default=30.0, help="Silence trimming threshold (dB)")
    p.add_argument("--pre-emphasis", type=float, default=0.0, help="Pre-emphasis coefficient (0 disables)")
    p.add_argument("--target-rms", type=float, default=0.1, help="Target RMS for normalization")
    p.add_argument("--noise-reduction", action="store_true", help="Enable spectral gating noise reduction")
    p.add_argument("--nr-n-fft", dest="nr_n_fft", type=int, default=1024)
    p.add_argument("--nr-hop-length", dest="nr_hop_length", type=int, default=256)
    p.add_argument("--nr-percentile", dest="nr_percentile", type=float, default=20.0)
    p.add_argument("--vad", action="store_true", help="Enable WebRTC VAD (voiced frames only)")
    p.add_argument("--vad-aggr", dest="vad_aggr", type=int, default=2, help="VAD aggressiveness 0-3")
    p.add_argument("--vad-frame-ms", dest="vad_frame_ms", type=int, default=20, help="VAD frame ms: 10/20/30")

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
    pi.add_argument("--top-k", type=int, default=0, help="Show top-k scores (0 disables)")
    pi.add_argument("--znorm", action="store_true", help="Apply simple z-norm to scores")
    pi.set_defaults(func=cmd_identify)

    # list
    pl = sub.add_parser("list", help="List enrolled speakers")
    pl.set_defaults(func=cmd_list)

    # remove
    pr = sub.add_parser("remove", help="Remove a speaker by name")
    pr.add_argument("name", type=str)
    pr.set_defaults(func=cmd_remove)

    # diarize
    pd = sub.add_parser("diarize", help="Naive diarization over sliding windows")
    pd.add_argument("audio", type=str, help="Path to audio file (wav/mp3/etc)")
    pd.add_argument("--num-speakers", type=int, default=None, help="Optional number of speakers")
    pd.add_argument("--assign", action="store_true", help="Assign clusters to enrolled speakers (nearest template)")
    pd.set_defaults(func=cmd_diarize)

    # mic
    pm = sub.add_parser("mic", help="Identify from microphone for N seconds")
    pm.add_argument("--seconds", type=float, default=3.0, help="Recording duration in seconds")
    pm.add_argument("--threshold", type=float, default=0.55, help="Cosine similarity threshold for unknown")
    pm.set_defaults(func=cmd_mic)

    pml = sub.add_parser("mic-live", help="Identify from microphone continuously")
    pml.add_argument("--window", type=float, default=1.5, help="Sliding window seconds")
    pml.add_argument("--hop", type=float, default=0.5, help="Hop seconds")
    pml.add_argument("--threshold", type=float, default=0.55, help="Cosine similarity threshold for unknown")
    pml.set_defaults(func=cmd_mic_live)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
