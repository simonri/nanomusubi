# autoresearch

This is an experiment to have the LLM do its own research on Wan 2.2 I2V LoRA training.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun14`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — latent and text encoder caching. Do not modify.
   - `wan_train_network.py` — the file you modify. Hyperparameters, LoRA config, timestep settings, optimizer, etc.
4. **Verify cache exists**: Check that `~/nanomusubi/runs/td/cache/` contains `.safetensors` cache files. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 15 minutes** (wall clock training time). You launch it as:

```
uv run accelerate launch --num_cpu_threads_per_process 4 src/musubi_tuner/wan_train_network.py \
    --dit ~/nous/comfyui-data/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
    --dataset_config ./runs/td/config.toml \
    --output_dir ./runs/td/output \
    --output_name wan2.2-i2v-high-td-v1.0 \
    --min_timestep 900
```

You may cancel the run before 10 minutes if you have seen enough (e.g. loss clearly diverging or not improving).

**What you CAN do:**
- Modify any file in the repo except `prepare.py`. This includes `wan_train_network.py`, `trainer_base.py`, model code, LoRA code, dataset code, etc.
- Add new packages via `uv add <package>`.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the caching logic and data loading.

**The goal is simple: get the lowest loss.** Since the time budget is fixed, you don't need to worry about training time — it's always up to 10 minutes. Lower loss means the model is learning the training videos better.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful loss gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes (or you cancel it) it prints a summary like this:

```
---
loss:             0.123456
training_seconds: 600.0
total_seconds:    640.0
peak_vram_mb:     45060.2
num_steps:        150
```

`training_seconds` is just the training loop time; `total_seconds` includes model loading.

You can extract the key metric from the log file:

```
grep "^loss:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. loss achieved (e.g. 0.123456) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 44.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	loss	memory_gb	status	description
a1b2c3d	0.123456	44.0	keep	baseline
b2c3d4e	0.118200	44.2	keep	increase LR to 2e-4
c3d4e5f	0.145000	44.0	discard	switch logsnr timestep sampling
d4e5f6g	0.000000	0.0	crash	network_dim=256 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/jun14`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `wan_train_network.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run accelerate launch ... > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^loss:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If loss improved (lower), you "advance" the branch, keeping the git commit
9. If loss is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take up to 15 minutes of training (+ a few minutes for model loading). If a run exceeds 25 minutes total, kill it and treat it as a failure (discard and revert).

**Early cancellation**: If you see the loss clearly diverging or stuck after a few minutes, you may cancel the run (Ctrl-C / kill) and treat it as a discard. Log `0.000000` loss with status `discard` and a note like "cancelled early — diverging".

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder. The loop runs until the human interrupts you, period.
