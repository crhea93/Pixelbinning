'''
DEFINE INPUTS IN MAIN FUNCTION AT END OF FILE
This file will create a bin map for each pixel. In reality we are simply creating a region about each pixel
with a predetermined signal-to-noise value.
---------------------------------------------------
---------------------------------------------------
Inputs:
    image_fits - Name of Image Fits File (e.g. "simple_image.fits")
    exposure_map - Name of exposure map -- optional -- (e.g. 'flux_broad.expmap')
    StN_Target - Target Signal-to-Noise (e.g. 50)
    pixel_size - Pixel radius in degrees (e.g. 0.492 for CHANDRA AXIS I)
    home_dir - Full Path to Home Directory (e.g. '/home/user/Documents/ChandraData/12833/repro')
    image_dir - Full Path to Image Directory (e.g. '/home/user/Desktop')
    output_dir - Full Path to Output Directory (e.g. '/home/user/Documents/ChandraData/12833')
---------------------------------------------------
---------------------------------------------------
Outputs:
    -- Creates bin plots for bin accretion and WVT algorithms
---------------------------------------------------
---------------------------------------------------
---------------------------------------------------
Carter Rhea
https://carterrhea.com
carterrhea93@gmail.com
'''

#-----------------INPUTS--------------------------#
import os
import gc
import sys
import time
import threading
import numpy as np
import statistics as stats
import multiprocessing as mp
from astropy.io import fits
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib as mpl
#-------------------------------------------------#
lock = threading.Lock()
#-------------------------------------------------#
# Bin Class Information
# Everything in here should be self-explanatory... if not let me know and I
# will most happily comment it! :)
class Bin:
    def __init__(self, number):
        self.bin_number = number
        self.pixels = []
        self.StN = [0]
        self.Signal = [0]
        self.Noise = [0]

    def add_pixel(self, Pixel):
        self.StN = 0
        self.pixels.append(Pixel)
        self.Signal[0] += Pixel.Signal
        self.Noise[0] += Pixel.Noise
        if self.Noise[0] != 0:
            self.StN = self.Signal[0]/(np.sqrt(self.Noise[0]))


#-------------------------------------------------#
#-------------------------------------------------#
# Pixel Class Information
# Ditto à propos the documentation for this class
class Pixel:
    def __init__(self, number, pix_x, pix_y, signal, noise):
        self.pix_number = number
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.Signal = signal
        self.Noise = noise
        self.StN = 0
        if self.Noise != 0:
            self.StN = self.Signal/np.sqrt(self.Noise)
        self.neighbors = []
        self.neighbors_x = []
        self.neighbors_y = []
        self.assigned_to_bin = False
        self.assigned_bin = None

#-------------------------------------------------#
#-------------------------------------------------#
# READ IN
# we first must read in our data from the Chandra file
# pixel (x,y) are CENTER coordinates!
#   parameters:
#       fits_file = fits file in string format
#       image_fits = fits image file in string format
#       image_coord = reg file containing center of box of interest and sizes
#       exposure_map = exposure map file in string format
def read_in(image_fits,exposure_map = 'none'):
    #Collect Pixel Data
    hdu_list = fits.open(image_fits, memmap=True)
    exposure_time = float(hdu_list[0].header["TSTOP"]) - float(hdu_list[0].header["TSTART"])
    counts = hdu_list[0].data
    y_len = counts.shape[0]
    x_len = counts.shape[1]
    hdu_list.close()
    x_min = 0; y_min = 0
    x_max = x_len; y_max = y_len
    if exposure_map.lower() != 'none':
        hdu_list = fits.open(exposure_map, memmap=True)
        exposure = hdu_list[0].data
        hdu_list.close()
    #Currently Do not bother reading background information
    '''bkg_hdu = fits.open(bkg_image_fits, memmap=True)
    bkg_counts = bkg_hdu[0].data
    bkg_hdu.close()
    avg_bkg_counts = np.mean(bkg_counts)
    bkg_sigma = np.std(bkg_counts)'''
    Pixels = []
    pixel_count = 0
    for col in range(int(x_len)):
        for row in range(int(y_len)):
            if exposure_map.lower() == 'none':
                flux = counts[row][col]
                vari = counts[row][col]
            else:
                flux = counts[row][col]/(exposure[row][col]*exposure_time) #- avg_bkg_counts/(exposure[row][col]*exposure_time)
                vari = counts[row][col]/(exposure[row][col]**2*exposure_time**2) #+ bkg_sigma
            Pixels.append(Pixel(pixel_count,x_min+col,y_min+row,flux,vari)) #Bottom Left Corner!
            pixel_count += 1
    #print("We have "+str(pixel_count)+" Pixels! :)")
    return Pixels, x_min, x_max, y_min, y_max
#-------------------------------------------------#
#-------------------------------------------------#
# CALCULATE NearestNeighbors
#   http://scikit-learn.org/stable/modules/neighbors.html
#   parameters:
#       pixel_list - list of pixel objects
def Nearest_Neighbors_ind(pixel_list,num_neigh,pix_id):
    xvals = []
    yvals = []
    for pixel in pixel_list:
        xvals.append(pixel.pix_x)
        yvals.append(pixel.pix_y)
    X = np.column_stack((xvals,yvals))
    nbrs = NearestNeighbors(n_neighbors=num_neigh, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    new_neighbors = []
    for j in range(num_neigh-1):
        index = indices[pix_id][j + 1]
        new_neighbors.append(pixel_list[index])
        '''if distances[pix_id][j+1] == 1:
            index = indices[pix_id][j+1]
            new_neighbors.append(pixel_list[index])
        else:
            pass #not adjacent'''
    return new_neighbors
#-------------------------------------------------#
#-------------------------------------------------#
# Create bin information for chandra
#   prameters:
#       Bins - list of final bins
#       min_x - minimum pixel x value
#       min_y - minimum pixel y value
#       output_directory - directory where txt file will be located
#       filename - name of text file
def Bin_data(Bins,Pixels,min_x,min_y, output_directory, filename):
    Bins.sort(key=lambda bin: bin.bin_number)
    file = open(output_directory+'/'+filename+'.txt',"w+")
    file.write("This text file contains information necessary for chandra to bin the pixels appropriately for image.fits \n")
    file.write("pixel_x pixel_y bin \n")
    file2 = open(output_directory + '/' + filename + '_bin.txt', "w+")
    file2.write(
        "This text file contains information necessary for chandra to bin the pixels appropriately for image.fits \n")
    file2.write("pixel_x pixel_y bin \n")
    binCount = 0
    for bin in Bins:
        #print(bin.pixels[0].pix_x)
        for pixel in bin.pixels:
            file.write(str(pixel.pix_x-min_x)+" "+str(pixel.pix_y-min_y)+" "+str(binCount)+' \n')
        binCount += 1
    binCount = 0
    for pixel_ in Pixels:
        file2.write(str(pixel_.pix_x-min_x)+" "+str(pixel_.pix_y-min_y)+" "+str(binCount)+'\n')
        binCount += 1
    file.close()
    file2.close()
    return None

#-------------------------------------------------#
#-------------------------------------------------#
# Signal_Complete Algorithm
# The goal of this subroutine is simply to fill each
# pixels neighbor list so that the required
# signal-to-noise is achieved
def Signal_Complete(pixel,Pixels,StN_Target,output_dir,out_name):
    Current_bin = Bin(pixel.pix_number)
    current_pixels = [pixel]
    #Current_bin.add_pixel(pixel)
    # Lets add pixels to the bin until the Signal to Noise is reached
    StN = pixel.StN
    StN_found = False
    neigh_num = 0
    i = 0  # how many times we have calculated nearest neighbors
    neighbors_unsearched = Nearest_Neighbors_ind(Pixels, 10 ** (i + 1), pixel.pix_number)
    while StN_found == False:
        #Current_bin.add_pixel(neighbors_unsearched[neigh_num])
        current_pixels.append(neighbors_unsearched[neigh_num])
        StN = neighbors_unsearched[neigh_num].StN
        if StN > StN_Target:
            StN_found = True
        neigh_num += 1
        if neigh_num >= len(neighbors_unsearched):
            i += 1  # So we can search further!
            neighbors_unsearched = Nearest_Neighbors_ind(Pixels, 10 ** (i + 1), pixel.pix_number)
    #print(Current_bin.pixels)
    with lock:
        file_out = open(output_dir+'/'+out_name+'.txt','a')
        for pixel_ in Current_bin.pixels:
            file_out.write(str(pixel.pix_number)+" "+str(pixel_.pix_x)+" "+str(pixel_.pix_y)+'\n')
        file_out.close()
    #print(sys.getrefcount(current_pixels))
    print("Thread")
    gc.collect()
    del Current_bin; del current_pixels; del neighbors_unsearched; del StN_found; del neigh_num; del i
    for name in dir():
        if name.startswith('Current') or name.startswith("current"):
            del globals()[name]

    return None
#-------------------------------------------------#
#-------------------------------------------------#
# Bin_Creation Algorithm Parallelized
#   parameters:
#       Pixels - list of unique pixels
#       StN_Target - Target value of Signal-to_noise
def Bin_Creation_Par(Pixels,StN_Target,output_dir,out_name,set_processes=1):
    #step 1:setup list of bin objects
    print("Starting Bin Accretion Algorithm")
    file_out = open(output_dir+'/'+out_name+".txt","w+")
    file_out.write("Bin Pixel_X Pixel_Y \n")
    file_out.close()
    pool = mp.Pool(processes=set_processes)
    results = [pool.map(Signal_Complete,args=(pixel,Pixels[:],StN_Target,output_dir,out_name,)) for pixel in Pixels[:]]
    #Bin_list = [p.get() for p in results]
    print("Completed Bin Accretion Algorithm")
    #print("There are a total of "+str(len(Bin_list))+" bins!")
    return None
#-------------------------------------------------#
#-------------------------------------------------#
#Read input file
#   parameters:
#       input file - .i input file
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def read_input_file(input_file,number_of_inputs):
    inputs = {}
    with open(input_file) as f:
        for line in f:
            if '=' in line:
                inputs[line.split("=")[0].strip().lower()] = line.split("=")[1].strip()
            else: pass
        if len(inputs) != number_of_inputs:
            print("Please recheck the input file since some parameter is missing...")
            print("Exiting program...")
            exit()
        else:
            print("Successfully read in input file")
            for key,val in inputs.items():
                if is_number(val) == True:
                    inputs[key] = float(val)
        return inputs
#-------------------------------------------------#
#-------------------------------------------------#
# #-------------------------------------------------#
#-------------------------------------------------#
def main():
    inputs = read_input_file(sys.argv[1],float(sys.argv[2]))
    os.chdir(inputs['home_dir'])
    if os.path.isdir(inputs['output_dir']+'/histograms') == False:
        os.mkdir(inputs['output_dir']+'/histograms')
    else:
        for graph in os.listdir(inputs['output_dir']+'/histograms'):
            if graph.endswith(".png"):
                os.remove(os.path.join(inputs['output_dir']+'/histograms', graph))
    for key,val in inputs.items():
        print("     "+key+" = "+str(val))
    print("#----------------Algorithm Part 1----------------#")
    Pixels,min_x,max_x,min_y,max_y = read_in(inputs['image_fits'],inputs['exposure_map'])
    start = time.time()
    Bin_Creation_Par(Pixels, inputs['stn_target'],inputs['output_dir'],"pix_data",int(inputs['num_processes']))
    print("The binning took %.2f seconds"%(float(time.time()-start)))
    print("#----------------Algorithm Complete--------------#")
    print("#----------------Information Stored--------------#")
main()
