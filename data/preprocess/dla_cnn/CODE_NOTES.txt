This file contains snippets of executables, notes essentially

stdbuf -o 0 python localize_model.py -i 10000000 -c '../models/training/model_gensample_v7.0' -r '../data/v7gensample/train_*' -e '../data/v7gensample/test_mix_23559_10000.npz' | tee ../tmp/stdout_train_7.0.txt
