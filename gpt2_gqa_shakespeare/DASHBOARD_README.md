# GPT-2 GQA Shakespeare — Dashboard add-on

## Run in Google Colab
1. Upload/extract this project ZIP into Google Drive.
2. In a Colab cell, change directory into the `gpt2_gqa_shakespeare` project folder.
3. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
4. Start the website:
   ```python
   %env GRADIO_SHARE=1
   !python dashboard.py
   ```
5. Open the `gradio.live` URL printed by Colab. It is temporary; keep it private and stop the notebook when finished.

## Run locally / on a server
```bash
pip install -r requirements.txt
python dashboard.py
```
Open `http://localhost:7860`. The dashboard uses port **7860** by default. To change it:
```bash
PORT=8080 python dashboard.py
```

## Features
- Project file inventory and training/validation token counts
- GPU model and memory, CPU and RAM snapshots
- Training start/stop controls, logs and train/validation loss graph
- Text generation from the saved checkpoint
- Import a compatible `.pt` checkpoint
- Export checkpoint and tokenizer as a ZIP

## Monitoring limitations and safety
- Press **Refresh Monitoring** to update status; this is not an always-on telemetry service.
- GPU/CPU/RAM readings are snapshots.
- The current training script starts a new model on each run; it does **not** resume training from a checkpoint.
- The dashboard reads the saved checkpoint's actual step and validation loss.
- Training logs are saved as `dashboard_training.log`.
- A Gradio share URL is temporary. For a permanent website, deploy on a server with persistent storage, authentication and HTTPS. Do not expose an unauthenticated training dashboard publicly.
