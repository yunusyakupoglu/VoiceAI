from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .embed import SpeakerEmbedder
from .store import SpeakerDatabase
from .audio import record_microphone

app = typer.Typer(add_completion=False, help="""
Konuşmacı tanıma: Kayıtlı kişileri seslerinden tanımlayın.

Komutlar:
- enroll: Bir kişiyi referans ses ile kaydet
- identify: Bir ses dosyasındaki konuşmacıyı tanı
- enroll-mic: Mikrofondan kayıt ile kişi kaydet
- identify-mic: Mikrofondan konuşmacıyı tanı
- list: Kayıtlı kişileri listele
- remove: Bir kişiyi veritabanından sil
- verify: İki sesin aynı kişiye ait olup olmadığını ölç
- reset: Veritabanını sıfırla
""")

console = Console()


DEFAULT_DB = str(Path.cwd() / "data" / "speakers.json")


def get_db(db_path: Optional[str]) -> SpeakerDatabase:
    path = db_path or DEFAULT_DB
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return SpeakerDatabase(path=path)


@app.command()
def enroll(
    name: str = typer.Argument(..., help="Kişi adı (ör. Ali)"),
    audio_path: str = typer.Argument(..., help="Referans ses dosyası (wav/mp3)"),
    db: Optional[str] = typer.Option(None, "--db", help=f"Veritabanı yolu (varsayılan: {DEFAULT_DB})"),
    device: Optional[str] = typer.Option(None, "--device", help="PyTorch cihazı (cuda/cpu)"),
):
    """Bir kişiyi referans ses ile kaydet."""
    embedder = SpeakerEmbedder(device=device)
    result = embedder.compute_embedding_from_file(audio_path)
    database = get_db(db)
    database.enroll(name, result.embedding)
    console.print(f"[green]Kayıt başarılı:[/green] {name}")


@app.command()
def identify(
    audio_path: str = typer.Argument(..., help="Tanımlanacak ses dosyası"),
    db: Optional[str] = typer.Option(None, "--db", help=f"Veritabanı yolu (varsayılan: {DEFAULT_DB})"),
    threshold: float = typer.Option(0.6, "--threshold", min=0.0, max=1.0, help="Benzerlik eşiği (0-1)"),
    top_k: int = typer.Option(3, "--top-k", help="En yakın kaç kişiyi gösterelim?"),
    device: Optional[str] = typer.Option(None, "--device", help="PyTorch cihazı (cuda/cpu)"),
):
    """Bir ses dosyasındaki konuşmacıyı tanı."""
    embedder = SpeakerEmbedder(device=device)
    result = embedder.compute_embedding_from_file(audio_path)
    database = get_db(db)
    name, score, ranked = database.identify(result.embedding, threshold=threshold, top_k=top_k)

    table = Table(title="En yakın eşleşmeler")
    table.add_column("Kişi")
    table.add_column("Benzerlik", justify="right")
    for person, s in ranked:
        table.add_row(person, f"{s:.3f}")

    if name is None:
        console.print("[yellow]Eşik altında - konuşmacı bilinmiyor.[/yellow]")
    else:
        console.print(f"[green]Tahmin:[/green] {name} (skor={score:.3f})")

    console.print(table)


@app.command("list")
def list_speakers(
    db: Optional[str] = typer.Option(None, "--db", help=f"Veritabanı yolu (varsayılan: {DEFAULT_DB})"),
):
    """Kayıtlı kişileri listele."""
    database = get_db(db)
    speakers = database.list_speakers()
    if not speakers:
        console.print("[yellow]Hiç kişi bulunamadı.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="Kayıtlı Kişiler")
    table.add_column("Ad")
    for s in speakers:
        table.add_row(s)
    console.print(table)


@app.command()
def remove(
    name: str = typer.Argument(..., help="Silinecek kişi adı"),
    db: Optional[str] = typer.Option(None, "--db", help=f"Veritabanı yolu (varsayılan: {DEFAULT_DB})"),
):
    """Bir kişiyi veritabanından sil."""
    database = get_db(db)
    ok = database.remove_speaker(name)
    if ok:
        console.print(f"[green]Silindi:[/green] {name}")
    else:
        console.print(f"[yellow]Bulunamadı:[/yellow] {name}")


@app.command()
def verify(
    audio_a: str = typer.Argument(..., help="Ses dosyası A"),
    audio_b: str = typer.Argument(..., help="Ses dosyası B"),
    device: Optional[str] = typer.Option(None, "--device", help="PyTorch cihazı (cuda/cpu)"),
):
    """İki sesin aynı kişiye ait olma benzerliğini döndür."""
    embedder = SpeakerEmbedder(device=device)
    emb_a = embedder.compute_embedding_from_file(audio_a).embedding
    emb_b = embedder.compute_embedding_from_file(audio_b).embedding
    # cosine similarity is dot product of L2-normalized vectors
    score = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9))
    console.print(f"Benzerlik (0-1): [bold]{score:.3f}[/bold]")


@app.command()
def reset(
    db: Optional[str] = typer.Option(None, "--db", help=f"Veritabanı yolu (varsayılan: {DEFAULT_DB})"),
):
    """Veritabanını sıfırla (tüm kayıtları siler)."""
    database = get_db(db)
    database.reset()
    console.print("[red]Veritabanı sıfırlandı.[/red]")


def main() -> None:  # for setuptools entry_points if needed
    app()


# === Live microphone commands ===

@app.command("enroll-mic")
def enroll_mic(
    name: str = typer.Argument(..., help="Kişi adı (ör. Ali)"),
    seconds: float = typer.Option(5.0, "--seconds", help="Kayıt süresi (sn)"),
    db: Optional[str] = typer.Option(None, "--db", help=f"Veritabanı yolu (varsayılan: {DEFAULT_DB})"),
    device: Optional[str] = typer.Option(None, "--device", help="PyTorch cihazı (cuda/cpu)"),
):
    """Mikrofondan kısa kayıt alıp kişiyi kaydet."""
    console.print(f"[cyan]{seconds} sn kayıt başlıyor. Konuşun...[/cyan]")
    waveform, sr = record_microphone(duration_seconds=seconds, sample_rate=16000, channels=1)
    console.print("[cyan]Kayıt bitti, işleniyor...[/cyan]")

    embedder = SpeakerEmbedder(device=device)
    result = embedder.compute_embedding_from_waveform(waveform)
    database = get_db(db)
    database.enroll(name, result.embedding)
    console.print(f"[green]Kayıt başarılı:[/green] {name}")


@app.command("identify-mic")
def identify_mic(
    seconds: float = typer.Option(5.0, "--seconds", help="Kayıt süresi (sn)"),
    db: Optional[str] = typer.Option(None, "--db", help=f"Veritabanı yolu (varsayılan: {DEFAULT_DB})"),
    threshold: float = typer.Option(0.6, "--threshold", min=0.0, max=1.0, help="Benzerlik eşiği (0-1)"),
    top_k: int = typer.Option(3, "--top-k", help="En yakın kaç kişiyi gösterelim?"),
    device: Optional[str] = typer.Option(None, "--device", help="PyTorch cihazı (cuda/cpu)"),
):
    """Mikrofondan kısa kayıt alıp konuşmacıyı tanı."""
    console.print(f"[cyan]{seconds} sn kayıt başlıyor. Konuşun...[/cyan]")
    waveform, sr = record_microphone(duration_seconds=seconds, sample_rate=16000, channels=1)
    console.print("[cyan]Kayıt bitti, işleniyor...[/cyan]")

    embedder = SpeakerEmbedder(device=device)
    result = embedder.compute_embedding_from_waveform(waveform)
    database = get_db(db)
    name, score, ranked = database.identify(result.embedding, threshold=threshold, top_k=top_k)

    table = Table(title="En yakın eşleşmeler")
    table.add_column("Kişi")
    table.add_column("Benzerlik", justify="right")
    for person, s in ranked:
        table.add_row(person, f"{s:.3f}")

    if name is None:
        console.print("[yellow]Eşik altında - konuşmacı bilinmiyor.[/yellow]")
    else:
        console.print(f"[green]Tahmin:[/green] {name} (skor={score:.3f})")

    console.print(table)
