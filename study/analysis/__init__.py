"""Preregistered analysis pipeline (plan §6/§7/§9; PREREG_ADDENDUM §7/§10).

Cell-addressed over study/runs/raw/{judge}/{task}_{split}/{fw}_run{r}.jsonl.
Entry point: study/analysis/run_all.py --split {dev,test} [--full].

Label access goes exclusively through study/analyze_study.py's guarded
loaders: dev via the tag-gated carve-out, test via the FREEZE-manifest gate.
This package never opens data/labels_hidden/ itself.
"""
