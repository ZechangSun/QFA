#!/usr/bin/env python

"""
Script to generate a PDF of desired sightline
Requires specdb for the spectral data
"""

import pdb

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(
        description='Analyze the desired sightline and generate a PDF (v1.0)')
    parser.add_argument("plate", type=int, help="Plate")
    parser.add_argument("fiber", type=int, help="Fiber")
    parser.add_argument("survey", type=str, help="SDSS_DR7, DESI_MOCK")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args=None):
    from pkg_resources import resource_filename
    from dla_cnn.data_model.sdss_dr7 import process_catalog_dr7
    from dla_cnn.data_model.desi_mocks import process_catalog_desi_mock

    if args is None:
        pargs = parser()
    else:
        pargs = args
    default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")
    if pargs.survey == 'SDSS_DR7':
        process_catalog_dr7(kernel_size=400, model_checkpoint=default_model,
                            output_dir="./", pfiber=(pargs.plate, pargs.fiber),
                            make_pdf=True)
    elif pargs.survey == 'DESI_MOCK':
        process_catalog_desi_mock(kernel_size=400, model_checkpoint=default_model,
                            output_dir="./", pfiber=(pargs.plate, pargs.fiber),
                            make_pdf=True)
    #
    print("See predictions.json file for outputs")

# Command line execution
if __name__ == '__main__':
    args = parser()
    main(args)

