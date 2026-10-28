#!/bin/sh
# Stage B runs: validator OFF vs ON on example/practice/dev, ON (full stack) on test. Sequential.
set -e
PY=.venv/bin/python
run() { $PY extract.py --inbox "$1" --output "out/$2.json" $3 2> "out/$2.log"; echo "ran $2"; }
run example     ex_off   --no-validate
run example     ex_on
run messages    pr_off   --no-validate
run messages    pr_on
run eval/injection/dev  dev_off --no-validate
run eval/injection/dev  dev_on
run eval/injection/test test_on
echo STAGEB_RUNS_DONE
