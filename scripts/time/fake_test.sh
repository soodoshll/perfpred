source faketorch/bin/activate
source init.sh
LD_PRELOAD=./fake_libcudart.so python -m perfpred.trace trace $@ 2> err.log
source torch_env/bin/activate
source init.sh
python -m perfpred.trace predict 2>> err.log