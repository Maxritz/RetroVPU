@echo off
cd /d %~dp0
git add -A
git commit -m "Add benchmark results, run script, scrub hardcoded paths

- benchmark_results.json: fresh 5-variant results (18.6-19.8x speedup range)
- build.bat: require OPENVINO_ROOT env var instead of hardcoded path
- run.bat: new run-only script for reproducing results
- README.md: updated with latest measured results and variance note
- .gitignore: keep benchmark results tracked, quarantine stays local

Clean-Room-Source: arXiv 0710.2442, section 5; arXiv 0910.2324, section 3
Clean-Room-Exposure: none"
git push origin master
echo "DONE"