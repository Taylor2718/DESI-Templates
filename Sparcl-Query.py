#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:49:47 2023

@author: taylo
"""
from sparcl.client import SparclClient
from dl import queryClient as qc
import pandas as pd

print("Reading DESI EDR Meta Data")

client = SparclClient()
query = """
SELECT zp.targetid, zp.survey, zp.program, zp.healpix,  
       zp.z, zp.zwarn, zp.coadd_fiberstatus, zp.spectype, 
       zp.mean_fiber_ra, zp.mean_fiber_dec, zp.zcat_nspec, 
       CAST(zp.zcat_primary AS INT), zp.desi_target,
       zp.sv1_desi_target, zp.sv2_desi_target, zp.sv3_desi_target
FROM desi_edr.zpix AS zp
"""
zpix = qc.query(sql = query, fmt='pandas')

zpix.to_parquet('Data/zpix_data.parquet', index=False)  # Save the DataFrame to a Parquet file

print("Completed reading DESI EDR Meta Data")
print("Data now saved as parquet file with name 'zpix_data.parquet' under Data folder.")
