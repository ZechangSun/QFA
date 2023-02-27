#/bin/bash
if [[ $1 == "" ]]; then echo "Usage: get_catalogs vX.X.X"; exit; fi
version=$1

# gensample 10k test DLAs
nice python -c "import data_loader as d; d.process_catalog_gensample(gensample_files_glob=\"../data/gensample_hdf5_files/test_dlas_96629_10000.hdf5\", json_files_glob=\"../data/gensample_hdf5_files/test_dlas_96629_10000.json\", output_dir=\"../tmp/model_${version}_data_testdlas10k96629/\")"

# gensample 5k test DLAs
nice python -c "import data_loader as d; d.process_catalog_gensample(gensample_files_glob=\"../data/gensample_hdf5_files/test_dlas_96451_5000.hdf5\", json_files_glob=\"../data/gensample_hdf5_files/test_dlas_96451_5000.json\", output_dir=\"../tmp/model_${version}_data_testdlas5k96451/\")"

# gensample 10k mixed
nice python -c "import data_loader as d; d.process_catalog_gensample(gensample_files_glob=\"../data/gensample_hdf5_files/test_mix_23559_10000.hdf5\", json_files_glob=\"../data/gensample_hdf5_files/test_mix_23559_10000.json\", output_dir=\"../tmp/model_${version}_data_genmix23559/\")"

# DR5
nice python -c "import data_loader as d; d.process_catalog_dr7(output_dir=\"../tmp/model_${version}_data_dr5/\")"

# DR12
nice python -c "import data_loader as d; d.process_catalog_csv_pmf(output_dir=\"../tmp/model_${version}_data_dr12/\")"
