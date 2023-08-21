"""
Script to generate summary table for given DESI FITS files
"""
import argparse
from dla_cnn.desi.DesiMock import DesiMock
from dla_cnn.desi.training_sets import select_samples_50p_pos_neg
from dla_cnn.desi.preprocess import label_sightline
from dla_cnn.desi.training_sets import split_sightline_into_samples  
import csv
from os.path import exists,join
from os import remove


def generate_summary_table(sightlines, output_dir, mode = "w"):
    """
    Generate a csv file to store some necessary information of the given sightlines. The necessary information means the id, z_qso,
    s/n of thelymann forest part(avoid dlas+- 3000km/s), the wavelength range and corresponding pixel number of each channel. And the csv file's format is like:
    
    id(int)， z_qso(float), s2n(float), wavelength_start_b(float), wavelength_end_b(float), pixel_start_b(int), pixel_end_b(int), wavelength_start_r(float), wavelength_end_r(float), pixel_start_r(int), pixel_end_r(int), wavelength_start_z(float), wavelength_end_z(float), pixel_start_z(int), pixel_end_z(int),dlas_col_density(str),dlas_central_wavelength(str)

    "wavelength_start_b" means the start wavelength value of b channel, "wavelength_end_b" means the end wavelength value of b channel, "pixel_start_b" means the start pixel number of b channel, "pixel_end_b" means the end pixel number of b channel
    so do the other two channels.Besides, "dlas_col_density" means the col_density array of the sightline, and "dlas_central_wavelength" means the central wavelength array means the central wavelength array of the given sightline. Due to the uncertainty of the dlas' number, we chose to use str format to store the two arrays,
    each array is written in the format like "value1,value2, value3", and one can use `str.split(",")` to get the data, the column density and central wavelength which have the same index in the two arrays corrspond to the same dla.
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parameters:
    sightlines: list of `dla_cnn.data_model.Sightline.Sightline` object, the sightline contained should contain the all data of b,r,z channel, and shouldn't be rebinned,
    output_dir: str, where the output csv file is stored, its format should be "xxxx.csv",
    mode: str, possible values "w", "a", "w" means writing to the csv file directly(overwrite the previous content), "a" means adding more data to the csv file(remaining the previous content)
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    return:
    None
    
    """
    #the header of the summary table, each element's meaning can refer to above comment
    headers = ["id","z_qso","s2n","wavelength_start_b","wavelength_end_b","pixel_start_b","pixel_end_b","wavelength_start_r",\
               "wavelength_end_r","pixel_start_r","pixel_end_r","wavelength_start_z","wavelength_end_z","pixel_start_z","pixel_end_z","dlas_col_density","dlas_central_wavelength"]
    #open the csv file
    with open(output_dir, mode=mode,newline="") as summary_table:
        summary_table_writer = csv.DictWriter(summary_table,headers)
        if mode == "w":
            summary_table_writer.writeheader()
        for sightline in sightlines:
            #test if this sightline can be used for sample
            label_sightline(sightline, kernel=400, REST_RANGE=[900,1346], pos_sample_kernel_percent=0.3)
            flux,classification,offsets,column_density = split_sightline_into_samples(sightline, REST_RANGE=[900,1346], kernel=400)
            sample_masks=select_samples_50p_pos_neg(sightline,kernel=400)
            if sample_masks.size!=0:
                dlas_col_density = ""
                dlas_central_wavelength = ""
                for dla in sightline.dlas:
                    if dla.central_wavelength/(sightline.z_qso+1) <1220 and dla.central_wavelength/(sightline.z_qso+1)>912:
                        dlas_col_density += str(dla.col_density)+","
                        dlas_central_wavelength += str(dla.central_wavelength)+","
                if dlas_central_wavelength!="":
                    #for each sightline, read its information and write to the csv file
                    info = {"id":sightline.id, "z_qso":sightline.z_qso, "s2n": sightline.s2n, "wavelength_start_b":10**sightline.loglam[0],\
                        "wavelength_end_b":10**sightline.loglam[sightline.split_point_br-1],"pixel_start_b":0,"pixel_end_b":sightline.split_point_br-1,\
                        "wavelength_start_r":10**sightline.loglam[sightline.split_point_br],"wavelength_end_r":10**sightline.loglam[sightline.split_point_rz-1],\
                        "pixel_start_r":sightline.split_point_br,"pixel_end_r":sightline.split_point_rz-1,"wavelength_start_z":10**sightline.loglam[sightline.split_point_rz],\
                        "wavelength_end_z":10**sightline.loglam[-1],"pixel_start_z":sightline.split_point_rz,"pixel_end_z":len(sightline.loglam)-1}
                info["dlas_col_density"] = dlas_col_density[:-1]
                info["dlas_central_wavelength"] = dlas_central_wavelength[:-1]
                #write to the csv file
                summary_table_writer.writerow(info)

def write_summary_table(nums, version,path, output_path):
    """
    Directly read data from fits files and write the summary table, the summary table contains all available sightlines(dlas!=[] and z_qso>2.33) in the given fits files.
    -----------------------------------------------------------------------------------------------------------------------------------------
    parameters:
    nums: list, the given fits files' id, its elements' format is int, and one should make sure all fits files are available before invoking this funciton, otherwise some sightlines can be missed;
    version: int, the version of the data set we use, e.g. if the version is v9.16, then version = 16
    path: str, the dir of the folder which stores the given fits file, the folder's structure is like folder-fits files' id - fits files , if you are still confused, you can check the below code about read data from the fits file;
    output_path: str, the dir where the summary table is generated, and if there have been a summary table, then we will remove it and generate a new summary table;
    ------------------------------------------------------------------------------------------------------------------------------------------
    retrun:
    None
    """
    #if exists summary table before, remove it
    if exists(output_path):
        remove(output_path)
    def write_as_summary_table(num):
        """
        write summary table for a single given fits file, if there have been a summary table then directly write after it, otherwise create a new one
        ---------------------------------------------------------------------------------------------------------------------------------------------
        parameter:
        num: int, the id of the given fits file, e.g. 700
        ---------------------------------------------------------------------------------------------------------------------------------------------
        return:
        None
        """
        #read data from fits file
        file_path = join(path,str(num))
        spectra = join(file_path,"spectra-%i-%i.fits"%(version,num))
        truth = join(file_path,"truth-%i-%i.fits"%(version,num))
        zbest = join(file_path,"zbest-%i-%i.fits"%(version,num))
        spec = DesiMock()
        spec.read_fits_file(spectra,truth,zbest)
        sightlines = []
        
        for key in spec.data.keys():
            if spec.data[key]["z_qso"]>2.33 and spec.data[key]["DLAS"]!=[]:
                sightlines.append(spec.get_sightline(key))
        #generate summary table
        if exists(output_path):
            generate_summary_table(sightlines,output_path,"a")
        else:
            generate_summary_table(sightlines,output_path,"w")
    bad_files = [] #store the fits files with problems 
    #for each id in nums, invoking the `write_as_summary_table` funciton
    for num in nums:
        #try:
        write_as_summary_table(num)
        #except:
            #if have problems append to the bad_files
            #bad_files.append(num)
    #assert bad_files==[], "these fits files have some problems, check them please, fits files' id :%s"%str(bad_files)

def parser(options = None):
    parser = argparse.ArgumentParser(description = "read the given FITS files and generate the summary table")
    parser.add_argument("numbers",type=str,help="Numbers of FITS files, splited by , such as '700,701,702'")
    parser.add_argument("version",type=int,help="version of the given dataset")
    parser.add_argument("path", type=str, help= "the path of the FITS files' folder" )
    parser.add_argument("output_path",type=str,help="the path of the summary table")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args = None):
    if args is None:
        pargs = parser()
    else:
        pargs = args
    numbers = pargs.numbers.split(",")
    numbers = list(map(int,list(numbers)))
    write_summary_table(numbers,pargs.version,pargs.path,pargs.output_path)

# Command line execution
if __name__ == "__main__":
    args = parser()
    main(args)
