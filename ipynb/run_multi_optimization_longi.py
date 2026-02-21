import multiprocessing
import os
import argparse
import glob
import shutil
import time

label = 'lv'
def process_input(data_path, sub_id):
    command = f"python ../../voxel2mesh/hippo_lv_train_longi.py --data_path {data_path} --tag {label} --learning_rate 0.0005 --sub_id {sub_id} --gpu 0"
    print(command)
    import subprocess
    subprocess.run(command, shell=True)
def main():
    # time.sleep(2.5*60*60)

    filelist = glob.glob(f"/root/LV/LV/Neurips_LV/ADNI_age_tgt_lv_optim/*.pkl")
    filelist = filelist[:50]
    print(filelist)
    # print(filelist)python
    num_item = 10
    filelist = filelist
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
            sub_id = fold.split("/")[-1].split("_")[0]
            data_path = rf"{fold}"
            exist5000 = glob.glob(f"/root/LV/lv-parametric-modelling/ipynb/whole_brain_structure/edinburgh/{label}_scan5/out/{sub_id}/5000*.npy")
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