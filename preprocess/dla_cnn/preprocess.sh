#!/bin/bash
bdird=../data/gensample_hdf5_files/dlas
bdirs=../data/gensample_hdf5_files/slls
bdirh=../data/gensample_hdf5_files/high_nhi
bdirt=../data/gensample_hdf5_files
bout=../data/v71gensample

# DLAs x2
for f in ${bdird}/*.hdf5; do
        bname=$(basename ${f} .hdf5)
        echo "Processing: ${bdird}/${bname} to ${bout}/train_dla_${bname}_1"
        nice python -c "import data_loader as d; d.preprocess_gensample_from_single_hdf5_file(datafile=\"${bdird}/${bname}\", save_file=\"${bout}/train_dla_${bname}_1\")"
        echo "Processing: ${bdird}/${bname} to ${bout}/train_dla_${bname}_2"
        nice python -c "import data_loader as d; d.preprocess_gensample_from_single_hdf5_file(datafile=\"${bdird}/${bname}\", save_file=\"${bout}/train_dla_${bname}_2\")"
done

# High NHI x2
nice python -c "import data_loader as d; d.preprocess_gensample_from_single_hdf5_file(datafile=\"${bdirh}/${bname}\", save_file=\"${bout}/train_dla_${bname}_1\")"
nice python -c "import data_loader as d; d.preprocess_gensample_from_single_hdf5_file(datafile=\"${bdirh}/${bname}\", save_file=\"${bout}/train_dla_${bname}_2\")"

# Dual DLAs x2
echo "Processing: ${bout}/train_overlapdlas_1"
nice python -c "import data_loader as d; d.preprocess_overlapping_dla_sightlines_from_gensample(save_file=\"${bout}/train_overlapdlas_1\")"
echo "Processing: ${bout}/train_overlapdlas_2"
nice python -c "import data_loader as d; d.preprocess_overlapping_dla_sightlines_from_gensample(save_file=\"${bout}/train_overlapdlas_2\")"

# SLLS x1
for f in ${bdirs}/*.hdf5; do
        bname=$(basename ${f} .hdf5)
        echo "Processing: ${bdirs}/${bname} to ${bout}/train_slls_${bname}_1"
        nice python -c "import data_loader as d; d.preprocess_gensample_from_single_hdf5_file(datafile=\"${bdirs}/${bname}\", save_file=\"${bout}/train_slls_${bname}\")"
done

# Test files
nice python -c "import data_loader as d; d.preprocess_gensample_from_single_hdf5_file(datafile=\"${bdirt}/test_dlas_96451_5000\", save_file=\"${bout}/test_dlas_96451_5000\")"
nice python -c "import data_loader as d; d.preprocess_gensample_from_single_hdf5_file(datafile=\"${bdirt}/test_dlas_96629_10000\", save_file=\"${bout}/test_dlas_96629_10000\")"
nice python -c "import data_loader as d; d.preprocess_gensample_from_single_hdf5_file(datafile=\"${bdirt}/test_mix_23559_10000\", save_file=\"${bout}/test_mix_23559_10000\")"
nice python -c "import data_loader as d; d.preprocess_gensample_from_single_hdf5_file(datafile=\"${bdirt}/test_slls_94047_5000\", save_file=\"${bout}/test_slls_94047_5000\")"
