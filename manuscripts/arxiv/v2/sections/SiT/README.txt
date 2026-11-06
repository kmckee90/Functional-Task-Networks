Build from any directory:
  bash /home/klmckee/Desktop/test_actl/functional_task_networks/manuscripts/arxiv/v2/sections/SiT/build.sh

Include from the v2 main.tex document body:
  \input{sections/SiT/section.tex}

main.tex has not been changed. standalone.tex copies its current preamble.
The section uses only packages already present in main.tex.

Sources remain at:
/media/klmckee/Seagate Portable Drive/contents/FTN-SiT-final-results
Source data: final_results.csv, transfer.csv, numerical/protocol.json, numerical/audit.json; evaluation and model definitions cross-checked against experiments/sit/continual/{memory_budget,union11,model.py}.
All figures directly reference the external drive using absolute paths. The drive must be mounted at the same location to rebuild. The PDF itself is self-contained.
Diagnostic table uses unweighted task means; MSE change excludes the final task.

Provenance limitation: the current local memory_budget Python sources do not match the hashes listed in the archive protocol. The archived protocol and numerical outputs are the authoritative sources; local code was used only to clarify implementation terminology.
