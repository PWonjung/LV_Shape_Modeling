import multiprocessing
import os
import argparse
import glob
import shutil
import time

label = 'mid2_LBC'
def process_input(data_path, sub_id):
    command = f"python ../../voxel2mesh/hippo_lv_trs_train.py --data_path {data_path} --tag {label}  --learning_rate 0.0005 --sub_id {sub_id} --gpu 3 --lda 2 0 0 0 0 0 0"
    # pm, cf, edge, lap, norm_con, l2_vert, l2_norm
    
    print(command)
    import subprocess
    subprocess.run(command, shell=True)
def main():

    filelist = glob.glob(f"/root/LV/LV/Neurips_LV/LBC_age_tgt/*.pkl")
    filelist = filelist
    print(filelist)
    # print(filelist)python
    num_item = 10
    filelist = filelist[:300]
    # print(len(filelist))
    # print(filelist)
    # return
    for i in range(len(filelist)//num_item+1):
        processes = []                                                                                                                                       

        if (i+1)*num_item > len(filelist):
            folders = filelist[i*num_item:]
        else:
            folders = filelist[i*num_item:(i+1)*num_item]
            print(len(folders))
            print(folders)

        for fold in folders:
            sub_id = fold.split("/")[-1][:-4]
            print(sub_id)
            data_path = rf"{fold}"
            exist5000 = glob.glob(f"/root/LV/lv-parametric-modelling/ipynb/MICCAI-LV/results/{label}/out/{sub_id}/2000*.npy")

            if exist5000==[]:
            # print(f"{sub_id=}")
                p = multiprocessing.Process(target=process_input, args=(data_path, sub_id,))
                processes.append(p)
                p.start()
            else:
                print(f"{sub_id} is already exists.")
    
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()

# python C:\Users\v1wpark\Documents\Projects\lv-parametric-modelling\voxel2mesh\template_redistribute.py --data_path C:\Users\v1wpark\Documents\Projects\lv-parametric-modelling\ipynb\edinburgh\template_mesh_r.pkl --tag template_r  --learning_rate 0.0005 --sub_id template_r
# python ../../voxel2mesh/hippo_p2p.py --data_path /root/LV/LV/0124_hippo/synthseg_l/LBC360002_synthseg_l_opt_data.pkl --tag synthseg_l_new_pm  --learning_rate 0.0005 --sub_id LBC360002 --gpu 1 --lda 0 0.5 1500 5 1 1 1