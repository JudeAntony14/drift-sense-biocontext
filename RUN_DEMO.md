# RUN_DEMO.md — Exact steps to record the BioContext technical demo (Windows)

Follow these commands **in order**, from a fresh clone. Start your screen
recorder right before Step 5, then run the single demo command and let it
finish — everything after that is automatic.

Total time: setup ~2-5 minutes (once), demo run ~30-90 seconds.

---

## 0. Prerequisites

- **Python 3.9+** installed and on PATH. Check with:
  ```powershell
  python --version
  ```
  If this fails, install Python from https://www.python.org/downloads/
  (tick **"Add python.exe to PATH"** during install), then reopen your
  terminal.
- **Git** installed (or download the repo as a ZIP from GitHub instead of
  cloning — both work).

Open **PowerShell** (or Command Prompt) and continue below.

---

## 1. Clone the repository

```powershell
git clone https://github.com/JudeAntony14/drift-sense-biocontext.git
cd drift-sense-biocontext
```

## 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Your prompt should now start with `(.venv)`. If PowerShell blocks the
activation script with an execution-policy error, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs numpy, opencv-python-headless, scipy, matplotlib, pandas,
pytest. Takes about 1-2 minutes.

## 4. (Optional) Sanity-check the install

```powershell
python -m pytest tests/ -v
```

You should see `6 passed`. This step is optional for the recording, but
it's a fast, visible way to prove the code actually runs before the main
demo (nice for the video if you want it).

---

## 5. 🔴 START SCREEN RECORDING HERE

## 6. Run the demo

```powershell
python -m biocontext.demo
```

That's it — this is the single command to record. It runs sequentially and
needs no further input. As it runs, it will:

1. **Generate the procedural synthetic dataset** — a fresh DRAM-style
   Search Image + 10x-shrunk Reference Image pair — and print the ground
   truth coordinates and decoy count to the terminal.
2. **Save and open** `demo_output/01_search_image.png` and
   `demo_output/02_reference_image.png`.
3. **Run the classical baseline** and save/open
   `demo_output/03_baseline_result.png` (prediction vs ground truth,
   candidate box drawn on the search image).
4. **Run BioContext** on the exact same pair and save/open
   `demo_output/04_biocontext_result.png` (candidates, false matches in
   red, final prediction vs ground truth).
5. **Save/open a 4-way comparison grid**
   `demo_output/05_method_comparison_grid.png` (baseline / retina-only /
   context-only / biocontext side by side on the same image).
6. **Run a real benchmark sweep** (~28-64 cases depending on flags) across
   all 4 methods and save/open comparison charts:
   `demo_output/figures/overall_comparison.png` and
   `demo_output/figures/error_by_factor.png`.
7. Print a final numeric recap to the terminal.

Each image auto-opens in your default viewer as soon as it's produced, so
just let the window come up on screen and keep it visible for a few
seconds before moving on — the terminal output narrates what you're
looking at in text, which doubles as on-screen captions for the recording.

If images don't auto-open on your system (or you'd rather open them
yourself between steps for more controlled pacing), use:

```powershell
python -m biocontext.demo --no-open
```

...and open each file from `demo_output/` manually in File Explorer as the
terminal prints its path.

To slow the pacing down (adds a pause after each step so you have time to
narrate or let the recording breathe):

```powershell
python -m biocontext.demo --pause 3
```

To run a fuller benchmark sweep (slower, more cases, slightly more
statistically solid numbers — good if you have 2-3 extra minutes):

```powershell
python -m biocontext.demo --benchmark-n 10
```

## 7. 🔴 STOP SCREEN RECORDING

Once the terminal prints:

```
STEP 7 / 7  --  Demo complete
```

...and the final headline numbers, you're done.

---

## 8. Where to find everything afterward

All demo output lives under `demo_output/` in the repo root:

```
demo_output/
  dataset/
    demo_case_search.png        # the exact generated Search Image
    demo_case_reference.png     # the exact generated Reference Image
    demo_case_meta.json         # ground-truth coordinates + full config, for reproducibility
  01_search_image.png           # labeled, presentation-ready Search Image
  02_reference_image.png        # labeled, upscaled Reference Image
  03_baseline_result.png        # baseline prediction vs ground truth
  04_biocontext_result.png      # BioContext prediction vs ground truth
  05_method_comparison_grid.png # 4-way side-by-side comparison
  tables/
    demo_benchmark_raw.csv      # every benchmark case x method, raw numbers
    demo_benchmark_summary.csv  # aggregated accuracy/success/runtime per method
    demo_benchmark_by_factor.csv
  figures/
    overall_comparison.png      # bar charts: mean error + success rate
    error_by_factor.png         # error broken down by stress factor
    success_by_factor.png
    runtime_comparison.png
    error_distribution.png
```

For the submission, the most important files to attach/screenshot alongside
the video are:
- `demo_output/05_method_comparison_grid.png` (the headline story in one image)
- `demo_output/figures/overall_comparison.png` (the benchmark numbers)
- `demo_output/tables/demo_benchmark_summary.csv` (raw numbers backing the claims)

The repository also ships a **pre-computed, larger benchmark run**
(`n_per_factor=10`, 64 cases) under `results/tables/` and `results/figures/`
at the repo root — these are the numbers quoted in the main `README.md`.
The `demo_output/` numbers from your recording will be close but not
bit-for-bit identical unless you pass the same `--benchmark-n`, since the
demo uses a smaller/faster sweep by default so it finishes quickly enough
to record live.

---

## 9. Re-running / troubleshooting

- **Re-run the whole demo fresh:** just run `python -m biocontext.demo`
  again — it overwrites `demo_output/` each time.
- **`ModuleNotFoundError`:** make sure `(.venv)` is active
  (`.venv\Scripts\activate`) and you ran `pip install -r requirements.txt`
  from inside the repo root.
- **Images don't auto-open:** use `--no-open` and open them manually from
  `demo_output/` in File Explorer — everything is still saved correctly.
- **Want a specific reproducible headline case:** the default seed (116)
  is pre-verified to reproduce the "baseline fooled by a repeated decoy,
  BioContext recovers via context" failure mode. To try a different one:
  ```powershell
  python -m biocontext.demo --seed 23
  ```
  (any integer works; not every seed produces a dramatic baseline failure,
  since most cases are genuinely unambiguous — that's expected and honest).
