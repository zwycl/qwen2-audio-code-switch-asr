# Reinforcement Learning for Sample-Efficient Code-Switched ASR

Code for the paper **"Reinforcement Learning for Sample-Efficient Code-Switched ASR"**

We adapt the audio-language model **Qwen2-Audio-7B-Instruct** to code-switched ASR with
**RLVR (reinforcement learning with verifiable rewards) using GRPO**. The recipe combines:

- a **CER reward** that directly optimizes transcription quality and eliminates whole-utterance translation errors,
- a **Script-Fidelity (SHR) reward** that penalizes characters from writing systems unrelated to the language pair, reducing script contamination, and
- a **two-pass draft-and-refinement** GRPO procedure ("listen again and fix") at training time.

Trained only on TTS-synthesized code-switched speech (CS-FLEURS), RLVR with **10% of the data
matches LoRA SFT trained on the full set**, with the largest gains on typologically distant pairs,
and the improvements **transfer zero-shot** to the human-recorded SwitchLingua corpus.

The training/RL code lives under `r1-aqa-main/` and is built on the
[R1-AQA](https://github.com/xiaomi-research/r1-aqa) GRPO framework.

## Repository layout

```
r1-aqa-main/
  run_csfleurs.sh                 # RLVR/GRPO training (-> src/train_csfleurs.py, DeepSpeed ZeRO-2)
  run_lora_csfleurs.sh            # LoRA SFT baseline (-> train_lora_csfleurs.py)
  run_eval_csfleurs.sh            # eval on CS-FLEURS read_test
  run_eval_switchlingua.sh        # zero-shot eval on SwitchLingua
  run_eval_multi_seed_*.sh        # multi-seed (n=3) eval used in the paper
  configs/zero2.json              # ZeRO-2 config for GRPO
  configs/lora_zero2.json         # ZeRO-2 config for LoRA
  src/
    train_csfleurs.py             # GRPO training entry (CER + SHR rewards, two-pass refinement)
    evaluate_csfleurs.py          # CER / SHR evaluation on CS-FLEURS
    evaluate_switchlingua.py      # CER / SHR evaluation on SwitchLingua
    preprocess_csfleurs_markers.py# strips code-switch markers / normalizes CS-FLEURS text
    utils/rewards.py              # CER, Script-Fidelity (SHR), and format rewards
    dataset/                      # csfleurs_dataset.py, switchlingua_dataset.py, vad_chunking.py
    trainer/grpo_trainer.py       # GRPO trainer
  train_lora_csfleurs.py          # LoRA SFT training script
  scripts/
    generate_fig_failure_modes.py # paper figure: translation / wrong-script rates
    generate_tab_lora_vs_rlvr.py  # paper table: LoRA vs RLVR win/loss analysis
eval_audio/                       # vendored text normalizers (whisper_normalizer/, cn_tn.py),
                                  # imported by utils/rewards.py for CER normalization
requirements_lora.txt             # LoRA-baseline dependencies
```

## Setup

```bash
pip install -r r1-aqa-main/requirements.txt
pip install -r requirements_lora.txt   # for the LoRA baseline
```

> Note: `src/utils/rewards.py` imports `whisper_normalizer` and (optionally) `cn_tn` for
> English / Chinese text normalization. Vendored copies are kept under `eval_audio/`; ensure they
> are importable (or `pip install whisper-normalizer`) before training/evaluating.

## Data

- **CS-FLEURS** — `XTTS-Train` (synthetic) for training, `Read-Test` (human-recorded) for
  evaluation. Run `python r1-aqa-main/src/preprocess_csfleurs_markers.py` first
  (see `run_eval_csfleurs.sh`).
- **SwitchLingua** — human-recorded code-switching, used for zero-shot transfer. We evaluate the
  8 pairs supported by Qwen2-Audio (ara, cmn, deu, fra, ita, jpn, kor, rus).

## Training

```bash
cd r1-aqa-main
bash run_csfleurs.sh          # RLVR / GRPO (CER + SHR reward, two-pass refinement)
bash run_lora_csfleurs.sh     # LoRA SFT baseline
```

Defaults follow the paper: Qwen2-Audio-7B-Instruct, frozen audio encoder, DeepSpeed ZeRO-2 on
8 GPUs, G=8 samples/prompt, lr 1e-6. Edit the data fraction / reward flags inside the scripts.

## Evaluation

```bash
cd r1-aqa-main
bash run_eval_csfleurs.sh         # CER + SHR on CS-FLEURS read_test
bash run_eval_switchlingua.sh     # CER + SHR on SwitchLingua (zero-shot)
# multi-seed (n=3) results as reported in the paper:
bash run_eval_multi_seed_csfleurs.sh
bash run_eval_multi_seed_switchlingua.sh
```

Metrics: **CER** (macro- and micro-averaged) and **SHR** (Script Hallucination Rate).

## Reproducing paper figures/tables

```bash
python r1-aqa-main/scripts/generate_fig_failure_modes.py   # translation / wrong-script rates
python r1-aqa-main/scripts/generate_tab_lora_vs_rlvr.py    # LoRA-vs-RLVR breakdown
```

## Acknowledgements

Built on [R1-AQA](https://github.com/xiaomi-research/r1-aqa) and
[Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio).
