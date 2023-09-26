#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:57:26 2023

@author: Taylor Pomfret
"""
import numpy as np
import pandas as pd
from astropy.io import fits
import os
from desitarget.sv1 import sv1_targetmask    # For SV1
from desitarget.sv2 import sv2_targetmask    # For SV2
from desitarget.sv3 import sv3_targetmask    # For SV3


class SpectraProcessor:
    def __init__(self, surveys_to_process, programs_to_process, 
                 objects_to_process, spectype_to_process, 
                 z_min_cond_to_process = 0, z_max_cond_to_process = 6,
                 number_of_spec = 1, healpix_to_process = 50, 
                 data_directory = "/Volumes/DESI-Data"):
        self.surveys = surveys_to_process
        self.programs = programs_to_process
        self.objects = objects_to_process
        self.spectype = spectype_to_process
        self.healpix = healpix_to_process
        self.specnum = number_of_spec
        self.z_min_cond = z_min_cond_to_process
        self.z_max_cond = z_max_cond_to_process
        self.data_directory = data_directory

    def load_desi(self):
        #self.lam = np.arange(3600, 9800, 0.8)
        zpix = pd.read_parquet('Data/zpix_data.parquet')  # Load the DESI Meta DataFrame from the Parquet file
        
        is_primary = zpix['zcat_primary']==1

        zpix_cat = zpix[is_primary]
        
        sv1_desi_tgt = zpix_cat['sv1_desi_target']
        sv2_desi_tgt = zpix_cat['sv2_desi_target']
        sv3_desi_tgt = zpix_cat['sv3_desi_target']
        
        ## DESI Bitmasks
        sv1_desi_mask = sv1_targetmask.desi_mask
        sv2_desi_mask = sv2_targetmask.desi_mask
        sv3_desi_mask = sv3_targetmask.desi_mask

        ## Candidate selection
            
        is_bgs = (sv1_desi_tgt & sv1_desi_mask['BGS_ANY'] != 0)|(sv2_desi_tgt & sv2_desi_mask['BGS_ANY'] != 0)|(sv3_desi_tgt & sv3_desi_mask['BGS_ANY'] != 0) #bright-galaxy-survey
        is_lrg = (sv1_desi_tgt & sv1_desi_mask['LRG'] != 0)|(sv2_desi_tgt & sv2_desi_mask['LRG'] != 0)|(sv3_desi_tgt & sv3_desi_mask['LRG'] != 0) #luminous-red-galaxy
        is_elg = (sv1_desi_tgt & sv1_desi_mask['ELG'] != 0)|(sv2_desi_tgt & sv2_desi_mask['ELG'] != 0)|(sv3_desi_tgt & sv3_desi_mask['ELG'] != 0) #emission-line-galaxy
        is_qso = (sv1_desi_tgt & sv1_desi_mask['QSO'] != 0)|(sv2_desi_tgt & sv2_desi_mask['QSO'] != 0)|(sv3_desi_tgt & sv3_desi_mask['QSO'] != 0) #quasar
        is_mws = (sv1_desi_tgt & sv1_desi_mask['MWS_ANY'] != 0)|(sv2_desi_tgt & sv2_desi_mask['MWS_ANY'] != 0)|(sv3_desi_tgt & sv3_desi_mask['MWS_ANY'] != 0) #milky-way-stars
        is_scnd = (sv1_desi_tgt & sv1_desi_mask['SCND_ANY'] != 0)|(sv2_desi_tgt & sv2_desi_mask['SCND_ANY'] != 0)|(sv3_desi_tgt & sv3_desi_mask['SCND_ANY'] != 0) #secondary-targets
        
        if self.objects == 'bgs':
            z_object = zpix_cat[is_bgs]
        elif self.objects == 'lrg':
            z_object = zpix_cat[is_lrg]
        elif self.objects == 'elg':
            z_object = zpix_cat[is_elg]
        elif self.objects == 'qso':
            z_object = zpix_cat[is_qso]
        elif self.objects == 'mws':
            z_object = zpix_cat[is_mws]
        elif self.objects == 'scnd':
            z_object = zpix_cat[is_scnd]
        else:
            raise Exception("Enter a valid object code.")
        
        filter_cond = ((z_object['spectype']==self.spectype) & 
                       (z_object['zcat_nspec']>=self.specnum) &
                       (z_object['survey'] == self.surveys[0]) & 
                       (z_object['program'] == self.programs[0]) & 
                       (z_object['z'] >= self.z_min_cond) &
                       (z_object['z'] <= self.z_max_cond)) #3 Spec Types: STAR, QSO, GALAXY

        z_object = z_object[filter_cond]
        z_object = z_object.sort_values('healpix')
        z_object = z_object.sort_values('program')
        total_unique_healpix_numbers = z_object['healpix'].unique()
        unique_healpix_numbers = z_object['healpix'].unique()[:self.healpix]

        # Filter the DataFrame to include only the rows with the selected unique healpix numbers
        self.z_object_filtered = z_object[z_object['healpix'].isin(unique_healpix_numbers)]
        self.z_object_filtered = self.z_object_filtered.sort_values('healpix')

        self.healpix_values = self.z_object_filtered['healpix'].unique()
        
        self.healpix_strings = [str(healpix) for healpix in self.healpix_values]
        self.healpix_shortcodes = []

        for healpix in self.healpix_values:
            if len(str(healpix)) == 3:
                shortcode = str(healpix)[:1]
            elif len(str(healpix)) == 4:
                shortcode = str(healpix)[:2]
            elif len(str(healpix)) >= 5:
                shortcode = str(healpix)[:3]

            if shortcode not in self.healpix_shortcodes:
                self.healpix_shortcodes.append(shortcode)

        sorted_programs = self.programs  # Programs order
        sorted_surveys = self.surveys  # Surveys order
        # Sort the DataFrame by 'survey', 'program', and 'healpix' columns
        self.z_object_filtered = self.z_object_filtered.sort_values(by=['survey', 'program', 'healpix'])

        # Reorder the rows based on the 'program' order defined in 'sorted_programs'
        self.z_object_filtered['program'] = pd.Categorical(self.z_object_filtered['program'], categories=sorted_programs, ordered=True)
        self.z_object_filtered = self.z_object_filtered.sort_values('program')

        # Reorder the rows based on the 'survey' order defined in 'sorted_surveys'
        self.z_object_filtered['survey'] = pd.Categorical(self.z_object_filtered['survey'], categories=sorted_surveys, ordered=True)
        self.z_object_filtered = self.z_object_filtered.sort_values('survey')


        # Now sort the DataFrame within each program and survey group by ascending 'healpix' numbers
        self.z_object_filtered = self.z_object_filtered.groupby(['survey', 'program']).apply(lambda x: x.sort_values('healpix'))

        # Reset the index of the resulting DataFrame
        self.z_object_filtered.reset_index(drop=True, inplace=True)

        self.filtered_id = list(self.z_object_filtered['targetid'])
        self.z = list(self.z_object_filtered['z'])
        
        available_healpix = len(total_unique_healpix_numbers)
        available_spectra = len(z_object)
        print(f"Number of healpix files available: {available_healpix}")
        print(f"Number of available spectra available: {available_spectra}")

        if self.healpix > available_healpix:
            raise Exception("Number of healpix files requested exceeds number of healpix files available")

    def process_files(self):
        # Specify the path to the directory you want to access
        mount_point = self.data_directory # The mount point of your hard drive
        directory_path = os.path.join(mount_point, "DESI-Data")

        # List all the files and directories in the specified path
        contents = os.listdir(directory_path)

        # Print the list of contents
        print("Surveys available in directory path:")
        for item in contents:
            print(item)
        
        def process_files_with_given_healpix(directory, surveys, programs, target_ids):
            lam = None
            flam_1 = []
            flam_2 = []
            flam_3 = []
            flam_1_noise = []
            flam_2_noise = []
            flam_3_noise = []
            flam_1_mask = []
            flam_2_mask = []
            flam_3_mask = []
            filtered_target_id = []

            for survey_dir in surveys:
                survey_path = os.path.join(directory, survey_dir)
                if not os.path.isdir(survey_path):
                    continue

                for program in programs:
                    program_path = os.path.join(survey_path, program)
                    if not os.path.isdir(program_path):
                        continue

                    for short_code in self.healpix_shortcodes:
                        short_code_path = os.path.join(program_path, short_code)
                        if not os.path.isdir(short_code_path):
                            continue

                        for healpix_dir in self.healpix_strings:
                            if not healpix_dir.isdigit():
                                continue
                            healpix_value = int(healpix_dir)
                            if healpix_value not in self.healpix_values:
                                continue
                            healpix_path = os.path.join(short_code_path, healpix_dir)
                            if not os.path.isdir(healpix_path):
                                continue

                            for file_name in os.listdir(healpix_path):
                                if file_name.endswith(".fits"):
                                    file_path = os.path.join(healpix_path, file_name)

                                    try:
                                        with fits.open(file_path, memmap=False) as hdulist:
                                            if lam is None:
                                                lam = (hdulist[3].data, hdulist[8].data, hdulist[13].data)
                                                
                                            target_ids_data = list(hdulist[1].data['targetid'])
                                            
                                            for i, target_id in enumerate(target_ids_data):
                                                if (target_id in target_ids) & (target_id not in filtered_target_id):
                                                    flam_1_data = hdulist[4].data #flux of b-channel spectra
                                                    flam_2_data = hdulist[9].data #flux of r-channel spectra
                                                    flam_3_data = hdulist[14].data #flux of z-channel spectra
                                                    
                                                    flam_1_noise_data = hdulist[5].data #noise flux of b-channel spectra
                                                    flam_2_noise_data = hdulist[10].data #noise flux of r-channel spectra
                                                    flam_3_noise_data = hdulist[15].data #noise flux of z-channel spectra
                                                    
                                                    flam_1_mask_data = hdulist[6].data #mask of b-channel spectra
                                                    flam_2_mask_data = hdulist[11].data #mask of r-channel spectra
                                                    flam_3_mask_data = hdulist[16].data #mask of z-channel spectra
                                                    
                                                    target_id_data = target_ids_data[i]
                                                    filtered_target_id.append(target_id_data)
                                                    
                                                    flam_1.append(flam_1_data[i])
                                                    flam_2.append(flam_2_data[i])
                                                    flam_3.append(flam_3_data[i])
                                                    
                                                    flam_1_noise.append(flam_1_noise_data[i])
                                                    flam_2_noise.append(flam_2_noise_data[i])
                                                    flam_3_noise.append(flam_3_noise_data[i])
                                                    
                                                    flam_1_mask.append(flam_1_mask_data[i])
                                                    flam_2_mask.append(flam_2_mask_data[i])
                                                    flam_3_mask.append(flam_3_mask_data[i])
                                                    
                                                    print(f"Succesfully processed object {target_id} from {file_name}")
                                                     

                                        flam = (flam_1, flam_2, flam_3)
                                        flam_noise = (flam_1_noise, flam_2_noise, flam_3_noise)
                                        flam_mask = (flam_1_mask, flam_2_mask, flam_3_mask)
                                        
                                    except IndexError:
                                        print(f"Skipping {file_name}, only coadded spectra files are processed")
                                        
            return lam, flam, flam_noise, flam_mask, filtered_target_id

        data_directory = "/Volumes/DESI-Data/desi-data"  # Replace with the actual directory path
        surveys_to_process = self.surveys # Specify the surveys to process
        programs_to_process = self.programs # Specify the programs to process
        target_ids_to_process = self.filtered_id
        self.lam, self.flam, self.flam_noise, self.flam_mask, self.target_ids = process_files_with_given_healpix(
            data_directory, surveys_to_process, programs_to_process, target_ids_to_process)
        #return self.lam, self.flam, self.flam_noise, self.flam_mask, self.target_ids
    
    def shift_and_normalize(self):
        #self.lam, self.flam, self.flam_noise, self.flam_mask, self.target_ids = self.process_files()
        self.p_lam = []
        self.p_flam = [] 
        self.p_flam_noise = [] 
        self.p_flam_mask = []

        for i in range(len(self.filtered_id)):
           
            self.lam_1 = self.lam[0]
            self.lam_2 = self.lam[1]
            self.lam_3 = self.lam[2]
            
            self.flam_1 = self.flam[0][i] #flux channels
            self.flam_2 = self.flam[1][i]
            self.flam_3 = self.flam[2][i]
            
            self.flam_1_noise = self.flam_noise[0][i] #noise channels
            self.flam_2_noise = self.flam_noise[1][i]
            self.flam_3_noise = self.flam_noise[2][i]
            
            self.flam_1_mask = self.flam_mask[0][i] #masking, 0=good
            self.flam_2_mask = self.flam_mask[1][i]
            self.flam_3_mask = self.flam_mask[2][i]
            
            overlap_1_start = np.searchsorted(self.lam_1, self.lam_2[0]) #lam_1
            overlap_1_end = np.searchsorted(self.lam_2, self.lam_1[-1]) #lam_2
            
            overlap_2_start = np.searchsorted(self.lam_2, self.lam_3[0]) #lam_2
            overlap_2_end = np.searchsorted(self.lam_3, self.lam_2[-1]) #lam_3
            
            combined_lam = np.concatenate((self.lam_1, self.lam_2[overlap_1_end:overlap_2_start], self.lam_3))
            combined_flam = np.concatenate((self.flam_1, self.flam_2[overlap_1_end:overlap_2_start], self.flam_3))
            combined_flam_noise = np.concatenate((self.flam_1_noise, self.flam_2_noise[overlap_1_end:overlap_2_start], self.flam_3_noise))
            combined_flam_mask = np.concatenate((self.flam_1_mask, self.flam_2_mask[overlap_1_end:overlap_2_start], self.flam_3_mask))
            
            self.p_lam.append(combined_lam)
            self.p_flam.append(combined_flam)
            self.p_flam_noise.append(combined_flam_noise)
            self.p_flam_mask.append(combined_flam_mask)
    


    def filter_zero_flux(self):
        
        self.p_lam = np.vstack(self.p_lam)
        self.p_flam = np.vstack(self.p_flam)
        self.p_flam_noise = np.vstack(self.p_flam_noise)
        self.p_flam_mask = np.vstack(self.p_flam_mask)

        non_zero_indices = np.all(self.p_flam != 0, axis=1)

        zero_counts = np.sum(self.p_flam == 0, axis=1)
        
        median_flux = np.median(self.p_flam, axis=1)

        hist, bin_edges = np.histogram(zero_counts, bins=[0, 2000, 4000, 6000, 8000])

        # Print the histogram results
        print("Histogram of zero counts:")
        for i in range(len(hist)):
            if i < len(hist) - 1:
                print(f"{bin_edges[i]}-{bin_edges[i+1]}: {hist[i]}")
            else:
                print(f"{bin_edges[i]} and above: {hist[i]}")

        non_median_indices = (np.where(median_flux!=0)[0])
        non_zero_count_indices = (np.where(zero_counts < 2000)[0])
        non_zero_indices = np.intersect1d(non_median_indices, non_zero_count_indices)
        
        self.p_lam = self.p_lam[non_zero_indices]
        self.p_flam = self.p_flam[non_zero_indices]
        self.p_flam_noise = self.p_flam_noise[non_zero_indices]
        self.p_flam_mask = self.p_flam_mask[non_zero_indices]
        
        median_flux = np.median(self.p_flam, axis=1, keepdims=True)
        self.p_flam /= median_flux #Normalised flux

        self.p_flam_noise *= (median_flux)**2
        max_noise = np.max(self.p_flam_noise, axis=1, keepdims=True)
        self.p_flam_noise /= max_noise #Normalised inverse variance

        target_id_df = pd.DataFrame({'targetid': self.target_ids})
        meta_unfiltered_df = pd.merge(target_id_df, self.z_object_filtered, on='targetid', how='inner')

        self.df_meta = meta_unfiltered_df.iloc[non_zero_indices] #Meta data
        self.df_meta = self.df_meta.reset_index(drop=True)
        
        self.z = np.array(self.df_meta['z'])
        self.p_lam /= (1+self.z[:, np.newaxis])

    def spectra_lam(self):
        spectra_length = len(self.df_meta)
        print(f"{spectra_length} spectra saved from {self.healpix} healpix files of the {self.programs[0]}, {self.surveys[0]} DESI survey")
        return self.p_lam

    def spectra_flam(self):
        return self.p_flam
    
    def spectra_noise(self):
        return self.p_flam_noise
    
    def spectra_mask(self):
        return self.p_flam_mask
    
    def meta(self):
        return self.df_meta
    
    def spectral_lines(self):
        
        df = pd.read_csv("Data/SDSS-Spec-Lines.csv")

        # Extract relevant columns
        wavelength = df.iloc[:, 0].values
        galaxy_weight = df.iloc[:, 1].values.astype(float)
        quasar_weight = df.iloc[:, 2].values.astype(float)
        absorption_emission_element = df.iloc[:, 3].values

        # Create the vstack
        spec_stack = np.vstack((wavelength, galaxy_weight, quasar_weight, absorption_emission_element))

        self.spec_stack_qso = np.delete(spec_stack[:, spec_stack[2] != 0], 1, axis=0)

        self.spec_stack_galaxy = np.delete(spec_stack[:, spec_stack[1] != 0], 2, axis=0)
        
        if self.spectype == 'QSO':
            return self.spec_stack_qso
        
        elif self.spectype == 'GALAXY':
            return self.spec_stack_galaxy
        
        else:
            return None


#%%
# Example usage in this file
if __name__ == "__main__":
    desi_data = SpectraProcessor(
        data_directory="/Volumes/DESI-Data",
        surveys_to_process=['sv1'],
        programs_to_process=['dark'],
        objects_to_process='mws',
        spectype_to_process='STAR',
        healpix_to_process = 3
    )
    desi_data.load_desi()
    desi_data.process_files()
    desi_data.shift_and_normalize()
    desi_data.filter_zero_flux()
    lam = desi_data.spectra_lam()
    flam = desi_data.spectra_flam()
    noise = desi_data.spectra_noise()
    mask = desi_data.spectra_mask()
    meta = desi_data.meta()
    