1) Loaded data from 4 subjects of the Calgary-Campinas dataset to the UHN cluster ("/cluster/projects/uludag/Brian/data/cc/train_3D/original")
- subject ID: 'rss_e14089s3_P53248.npy','rss_e14134s3_P06656.npy','rss_e14140s3_P52224.npy','rss_e14141s3_P58880.npy'
- coil sensitivies were estimated using BART ESPIRiT (ecalib)

2) Run gen_data_h4h.py to create 20 simulated motion cases per subject (total 80 motion simulations) with max motion extent of 7 mm and 7 deg
- simulated corrupted image volumes stored in /cluster/projects/uludag/Brian/data/cc/train_3D/corrupted
- NB. chose to save entire volumes separately, so that I can reuse this dataset for other purpopses

3) Run gen_slices.py to create training dataset (following Al-masni et al set-up) based on simulated motion-corrupted images stored in /cluster/projects/uludag/Brian/data/cc/train_3D/corrupted
- resulting dataset will be stored in /cluster/projects/uludag/Brian/data/cc/train_3D/corrupted/slices