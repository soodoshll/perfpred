rm err.log
#source faketorch/bin/activate
#source init.sh
source torch_env/bin/activate
source init.sh
python -m perfpred.trace trace $@ 2> err.log 
python -m perfpred.trace predict 2>> err.log > report.log
