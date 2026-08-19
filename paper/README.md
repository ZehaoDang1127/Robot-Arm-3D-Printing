# Robotic Printing Platform workshop paper

This folder contains a US-Letter, 10-point, two-column IROS workshop-style
LaTeX working draft about the functionality implemented in the repository as
of 7 August 2026. It uses the official RAS PaperCept `ieeeconf` class. The
absence of an abstract is intentional: the project is still accumulating the
perception, dynamic-surface, control, and experimental content needed for the
final paper.

Workshop page limits, anonymity, and archival policies are set by each
workshop. Before submission, compare this draft with the target workshop's call
for papers and replace the visible author placeholder. The official PaperCept
template and formatting guidance are available at
<https://ras.papercept.net/conferences/support/tex.php>.

## Files

- `main.tex` -- manuscript source.
- `references.bib` -- verified publication and documentation records.
- `ieeeconf.cls` -- official RAS PaperCept conference class, vendored so the
  workshop layout compiles reproducibly.
- `figures/ur5e_replay.png` -- representative frame from the repository's
  recorded Isaac Sim replay; it is qualitative evidence only.
- `data/compact_patch_metrics.csv` -- metrics from a fresh common-input run on
  Panda, UR5, and UR5e using the current working tree.
- `main.pdf` -- compiled manuscript.

## Reproduce the compact three-robot case

The common input is `../outputs/volumetric_patch.gcode`, whose positive `E`
values are cubic millimetres. Run Panda and UR5 together, then UR5e:

```powershell
python ../run_pipeline.py ../outputs/volumetric_patch.gcode `
  --material alginate_chitosan_pic_al1ch1_research `
  --robot both --lo 0 --hi 1 --max-seg-len-mm 1 `
  --simplify-deg 0 --ik-selection-mode greedy `
  --output-dir evaluation

python ../run_pipeline.py ../outputs/volumetric_patch.gcode `
  --material alginate_chitosan_pic_al1ch1_research `
  --robot ur5e --isaac-usd ../UR5e_extruder.usd `
  --lo 0 --hi 1 --max-seg-len-mm 1 `
  --simplify-deg 0 --ik-selection-mode greedy `
  --output-dir evaluation
```

## Compile

With a standard TeX Live installation, run from this folder:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript uses BibTeX with the standard `IEEEtran` bibliography style.
Author names and affiliations are deliberately left as a visible placeholder
for the research team to complete.
