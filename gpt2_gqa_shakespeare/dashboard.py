"""Gradio monitoring dashboard for the GPT-2 GQA Shakespeare project."""
from pathlib import Path
import os, sys, time, threading, subprocess, zipfile, tempfile
from datetime import datetime
import gradio as gr
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "best.pt"
DATA = ROOT / "data"
LOG = ROOT / "dashboard_training.log"
_lock = threading.Lock()
_process = None
_started_at = None

def fmt_bytes(n):
    if n is None: return "—"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB": return f"{n:.1f} {unit}"
        n /= 1024

def system_status():
    gpu = "Unavailable"
    gpu_mem = "—"
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            gpu_mem = f"{fmt_bytes(total-free)} used / {fmt_bytes(total)} total"
        except Exception:
            gpu = "CUDA available"
    try:
        import psutil
        cpu = f"{psutil.cpu_percent(interval=0.15):.0f}%"
        ram = f"{psutil.virtual_memory().percent:.0f}% ({fmt_bytes(psutil.virtual_memory().used)} / {fmt_bytes(psutil.virtual_memory().total)})"
    except Exception:
        cpu, ram = "Install psutil", "Install psutil"
    return gpu, gpu_mem, cpu, ram

def inspect_project():
    paths = {
        "Checkpoint": CKPT,
        "Tokenizer": DATA / "tokenizer.json",
        "Training IDs": DATA / "train_ids.npy",
        "Validation IDs": DATA / "val_ids.npy",
        "Training script": ROOT / "train.py",
        "Generation script": ROOT / "generate.py",
    }
    rows = []
    for name, p in paths.items():
        rows.append([name, "FOUND" if p.exists() else "MISSING", fmt_bytes(p.stat().st_size) if p.exists() and p.is_file() else "—"])
    train_n = val_n = "—"
    for p, target in [(DATA/"train_ids.npy","train"),(DATA/"val_ids.npy","val")]:
        if p.exists():
            try:
                n = len(np.load(p, mmap_mode="r"))
                if target == "train": train_n = f"{n:,}"
                else: val_n = f"{n:,}"
            except Exception: pass
    return rows, train_n, val_n

def checkpoint_summary():
    if not CKPT.exists():
        return "No checkpoint found yet."
    try:
        obj = torch.load(CKPT, map_location="cpu", weights_only=False)
        return (f"Checkpoint: `{CKPT.name}`  \n"
                f"Saved training step: **{obj.get('step', 'unknown')}**  \n"
                f"Best validation loss: **{obj.get('best_val_loss', 'unknown')}**  \n"
                f"File size: **{fmt_bytes(CKPT.stat().st_size)}**  \n"
                f"Last updated: **{datetime.fromtimestamp(CKPT.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}**")
    except Exception as e:
        return f"Checkpoint exists but could not be read: {e}"

def read_log():
    if not LOG.exists(): return "No training log yet. Start a training run to populate it."
    return LOG.read_text(errors="replace")[-12000:]

def parse_losses():
    steps, train, val = [], [], []
    if LOG.exists():
        import re
        for line in LOG.read_text(errors="replace").splitlines():
            m = re.search(r"step\s+(\d+)\s+\|\s+train\s+([0-9.]+)\s+\|\s+val\s+([0-9.]+)", line)
            if m:
                steps.append(int(m.group(1))); train.append(float(m.group(2))); val.append(float(m.group(3)))
    if not steps: return None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(steps, train, marker="o", label="Train loss")
    ax.plot(steps, val, marker="o", label="Validation loss")
    ax.set_xlabel("Training step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training / validation loss"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    return fig

def refresh():
    gpu, mem, cpu, ram = system_status()
    rows, tr, va = inspect_project()
    running = _process is not None and _process.poll() is None
    return (
        f"**{'RUNNING' if running else 'IDLE'}**",
        gpu, mem, cpu, ram, rows, tr, va,
        checkpoint_summary(), read_log(), parse_losses()
    )

def _train_worker(args):
    global _process
    cmd = [sys.executable, str(ROOT/"train.py"),
           "--max-iters", str(args["steps"]),
           "--eval-interval", str(args["eval_interval"]),
           "--eval-iters", str(args["eval_iters"]),
           "--batch-size", str(args["batch_size"]),
           "--block-size", str(args["block_size"]),
           "--learning-rate", str(args["learning_rate"]),
           "--device", "auto"]
    with LOG.open("w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n\n"); f.flush()
        try:
            _process = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
            _process.wait()
            f.write(f"\nDashboard process exit code: {_process.returncode}\n")
        except Exception as e:
            f.write(f"\nERROR: {e}\n")

def start_training(steps, eval_interval, eval_iters, batch_size, block_size, learning_rate):
    global _process
    with _lock:
        if _process is not None and _process.poll() is None:
            return "A training process is already running. Stop it before starting another."
        if not all((DATA/"tokenizer.json").exists(), (DATA/"train_ids.npy").exists(), (DATA/"val_ids.npy").exists()):
            return "Dataset/tokenizer files are missing. Run prepare.py first."
        args = dict(steps=int(steps), eval_interval=int(eval_interval), eval_iters=int(eval_iters),
                    batch_size=int(batch_size), block_size=int(block_size), learning_rate=float(learning_rate))
        threading.Thread(target=_train_worker, args=(args,), daemon=True).start()
    return "Training started. Use Refresh Monitoring to update the live status."

def stop_training():
    global _process
    if _process is not None and _process.poll() is None:
        _process.terminate()
        return "Stop signal sent. The latest checkpoint saved before stopping should remain available."
    return "No active training process."

def generate_text(prompt, max_new_tokens, temperature, top_k):
    if not CKPT.exists(): return "No checkpoint found. Train the model or import a checkpoint first."
    try:
        sys.path.insert(0, str(ROOT))
        from generate import load_model
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model, tokenizer, meta = load_model(CKPT, device)
        ids = tokenizer.encode(prompt or "ROMEO:")
        if not ids: ids = [32]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(x, int(max_new_tokens), float(temperature), int(top_k) if int(top_k)>0 else None)
        return tokenizer.decode(out[0].tolist())
    except Exception as e:
        return f"Generation error: {type(e).__name__}: {e}"

def export_checkpoint():
    if not CKPT.exists(): raise gr.Error("No checkpoint is available to export.")
    out = ROOT / "gpt2_gqa_checkpoint_export.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CKPT, "checkpoints/best.pt")
        tok = DATA/"tokenizer.json"
        if tok.exists(): z.write(tok, "data/tokenizer.json")
    return str(out)

def import_checkpoint(file):
    if file is None: return "Choose a .pt checkpoint file."
    src = Path(file)
    try:
        payload = torch.load(src, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload or "config" not in payload:
            return "Invalid checkpoint: expected a project checkpoint containing model and config."
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, CKPT)
        return "Checkpoint imported successfully. Refresh Monitoring to see its step and validation loss."
    except Exception as e:
        return f"Import failed: {type(e).__name__}: {e}"

with gr.Blocks(title="GPT-2 GQA Shakespeare | Monitor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GPT-2 GQA Shakespeare — Monitoring Console\nLive project health, training controls, loss tracking, checkpoint management and text generation.")
    with gr.Row():
        status = gr.Markdown("**IDLE**")
        refresh_btn = gr.Button("↻ Refresh Monitoring", variant="primary")
    with gr.Tab("Overview"):
        with gr.Row():
            gpu_box = gr.Textbox(label="GPU", interactive=False)
            gpu_mem_box = gr.Textbox(label="GPU memory", interactive=False)
            cpu_box = gr.Textbox(label="CPU usage", interactive=False)
            ram_box = gr.Textbox(label="RAM usage", interactive=False)
        with gr.Row():
            train_tokens = gr.Textbox(label="Training tokens", interactive=False)
            val_tokens = gr.Textbox(label="Validation tokens", interactive=False)
        inventory = gr.Dataframe(headers=["Component","Status","Size"], label="Project files", interactive=False)
        checkpoint_info = gr.Markdown()
    with gr.Tab("Training"):
        gr.Markdown("Training starts a new run using the existing `train.py`. This project version does **not** resume optimizer/model training from a prior checkpoint.")
        with gr.Row():
            steps = gr.Slider(20, 5000, value=500, step=20, label="Training steps")
            eval_interval = gr.Slider(10, 500, value=100, step=10, label="Evaluation interval")
            eval_iters = gr.Slider(5, 50, value=10, step=5, label="Evaluation batches")
        with gr.Row():
            batch_size = gr.Dropdown([4,8,16,32], value=16, label="Batch size")
            block_size = gr.Dropdown([48,64,128,256], value=128, label="Block size")
            learning_rate = gr.Number(value=0.0003, label="Learning rate")
        with gr.Row():
            start_btn = gr.Button("▶ Start Training", variant="primary")
            stop_btn = gr.Button("■ Stop Training", variant="stop")
        train_message = gr.Textbox(label="Training control", interactive=False)
        loss_plot = gr.Plot(label="Loss curve")
        log_box = gr.Textbox(label="Training log (latest)", lines=14, interactive=False)
        start_btn.click(start_training, [steps,eval_interval,eval_iters,batch_size,block_size,learning_rate], train_message)
        stop_btn.click(stop_training, outputs=train_message)
    with gr.Tab("Text Generation"):
        prompt = gr.Textbox(value="ROMEO:", label="Prompt")
        with gr.Row():
            new_tokens = gr.Slider(20,500,value=200,step=10,label="New tokens")
            temperature = gr.Slider(0.2,1.5,value=0.8,step=0.05,label="Temperature")
            top_k = gr.Slider(0,100,value=50,step=1,label="Top-k (0 = disabled)")
        gen_btn = gr.Button("Generate Shakespeare-style text", variant="primary")
        generated = gr.Textbox(label="Generated text", lines=16)
        gen_btn.click(generate_text,[prompt,new_tokens,temperature,top_k],generated)
    with gr.Tab("Import / Export"):
        gr.Markdown("Import a compatible project `.pt` checkpoint. Export creates a ZIP containing the checkpoint and tokenizer.")
        upload = gr.File(label="Import checkpoint (.pt)", file_types=[".pt"], type="filepath")
        import_btn = gr.Button("Import checkpoint")
        import_msg = gr.Textbox(label="Import result", interactive=False)
        import_btn.click(import_checkpoint, upload, import_msg)
        export_btn = gr.Button("Prepare checkpoint export", variant="primary")
        export_file = gr.File(label="Export ZIP")
        export_btn.click(export_checkpoint, outputs=export_file)
    refresh_btn.click(refresh, outputs=[status,gpu_box,gpu_mem_box,cpu_box,ram_box,inventory,train_tokens,val_tokens,checkpoint_info,log_box,loss_plot])
    demo.load(refresh, outputs=[status,gpu_box,gpu_mem_box,cpu_box,ram_box,inventory,train_tokens,val_tokens,checkpoint_info,log_box,loss_plot])

if __name__ == "__main__":
    # Colab: share=True provides a temporary public URL. Local/server: use port 7860.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT","7860")),
                share=os.environ.get("GRADIO_SHARE","0") == "1")
