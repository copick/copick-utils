"""
A minimal example using minimal libraries / imports to download relevant AreTomo files from the CryoET Data Portal. Downloads the corresponding files, using the run ID as the base filename.
"""
import multiprocessing, requests, os
import pandas as pd
import numpy as np
import mdocfile

import cryoet_data_portal as cdp
import s3fs

global_client = cdp.Client()

def download_aretomo_files(dataset_id: int, output_dir: str):
    print(f"Fetching tiltseries for dataset id {dataset_id}...", flush=True)
    tiltseries_list: list[cdp.TiltSeries] = [tiltseries for run in cdp.Dataset.get_by_id(global_client, dataset_id).runs for tiltseries in run.tiltseries] # a bit slow for some reason, can take some time
    tiltseries_run_ids_and_ts_ids = [(ts.run.id, ts.id) for ts in tiltseries_list]
    print(f"Found {len(tiltseries_run_ids_and_ts_ids)} tiltseries for dataset id {dataset_id}. Starting downloads...", flush=True)
    with multiprocessing.Pool(processes=8) as pool: # adjust number of processes as needed
        for _ in pool.imap_unordered(_worker_download_aretomo_files_for_tiltseries, [(dataset_id, run_name, output_dir, tiltseries_id) for run_name, tiltseries_id in tiltseries_run_ids_and_ts_ids]):
            pass
            
def _worker_download_aretomo_files_for_tiltseries(args):
    dataset_id, run_name, output_dir, tiltseries_id = args
    download_aretomo_files_for_tiltseries(dataset_id, run_name, output_dir, tiltseries_id)

# note: this function assumes that there is only one tiltseries per run
# note: the tiltseries name is equivlaent to the run name 
# if tiltseries_id is provided, will be prioritized over dataset_id + run_name
def download_aretomo_files_for_tiltseries(dataset_id: int, run_name: str, output_dir: str, tiltseries_id: int = None):

    print(f"[{run_name}] Downloading AreTomo files for tiltseries id {tiltseries_id}...", flush=True)

    client = cdp.Client()
    s3 = s3fs.S3FileSystem(anon=True)
    if not tiltseries_id:
        all_tiltseries = cdp.TiltSeries.find(client, query_filters=[cdp.TiltSeries.run.dataset_id == dataset_id, cdp.TiltSeries.run.name == run_name])
        if len(all_tiltseries) == 0:
            raise ValueError(f"No tiltseries found for dataset_id {dataset_id} and run_name {run_name}")
        if len(all_tiltseries) > 1:
            raise ValueError(f"Multiple tiltseries found for dataset_id {dataset_id} and run_name {run_name}")
        tiltseries = all_tiltseries[0]
    else:
        tiltseries = cdp.TiltSeries.get_by_id(client, tiltseries_id)

    # get the s3 folder path and then glob for *.tlt / *.rawtlt files to download them, renaming the base to match the run id
    s3_folder_path = tiltseries.s3_mrc_file.rsplit('/', 1)[0] + '/'
    tlt_files = s3.glob(s3_folder_path + '*.tlt') + s3.glob(s3_folder_path + '*.rawtlt')
    for tlt_file in tlt_files:
        base_name = os.path.basename(tlt_file)
        ext = os.path.splitext(base_name)[1]
        dest_file = os.path.join(output_dir, f"{tiltseries.run.id}{ext}")
        s3.get(tlt_file, dest_file)
        print(f"[{tiltseries.run.id}] Downloaded {base_name} as {os.path.basename(dest_file)}.", flush=True)

    # do the same for "*CTF*.txt" files and "*ctf*.txt" files
    ctf_files = s3.glob(s3_folder_path + '*CTF*.txt') + s3.glob(s3_folder_path + '*ctf*.txt')
    if len(ctf_files) == 0:
        print(f"WARNING: No CTF files found for tiltseries id {tiltseries.id}")
    else:
        ctf_file = ctf_files[0]
        base_name = os.path.basename(ctf_file)
        if len(ctf_files) > 1:
            print(f"WARNING: Multiple CTF files found for tiltseries id {tiltseries.id}, using {base_name}")
        ext = os.path.splitext(base_name)[1]
        dest_file = os.path.join(output_dir, f"{tiltseries.run.id}_CTF.txt")       
        s3.get(ctf_file, dest_file)
        print(f"[{tiltseries.run.id}] Downloaded {base_name} as {os.path.basename(dest_file)}.", flush=True)

    # now find the corresponding alignment for this tiltseries and download the "*.aln" file
    if len(tiltseries.alignments) == 0:
        print(f"WARNING: No alignments found for tiltseries id {tiltseries.id}")
    elif len(tiltseries.alignments) > 1:
        print(f"WARNING: Multiple alignments found for tiltseries id {tiltseries.id}")
    else:
        alignment = tiltseries.alignments[0]
        s3_alignment_folder_path = alignment.s3_alignment_metadata.rsplit('/', 1)[0] + '/'
        aln_files = s3.glob(s3_alignment_folder_path + '*.aln')
        if len(aln_files) == 0:
            raise ValueError(f"No .aln files found for run name {tiltseries.run.name} and alignment id {alignment.id}")
        aln_file = aln_files[0]
        base_name = os.path.basename(aln_file)
        if len(aln_files) > 1:
            print(f"WARNING: Multiple .aln files found for run name {tiltseries.run.name}, using {base_name}")
        ext = os.path.splitext(base_name)[1]
        dest_file = os.path.join(output_dir, f"{tiltseries.run.id}{ext}")
        s3.get(aln_file, dest_file)
        print(f"[{tiltseries.run.id}] Downloaded {base_name} as {os.path.basename(dest_file)}.", flush=True)

    # now get the mdoc file from the Frames/ folder
    frames = tiltseries.run.frames
    if len(frames) == 0:
        raise ValueError(f"No frames found for run name {tiltseries.run.name}")
    frame = frames[0]
    s3_frames_folder_path = frame.s3_frame_path.rsplit('/', 1)[0] + '/'
    mdoc_files = s3.glob(s3_frames_folder_path + '*.mdoc')
    if len(mdoc_files) == 0:
        raise ValueError(f"No .mdoc files found for run name {tiltseries.run.name}")
    mdoc_file = mdoc_files[0]
    base_name = os.path.basename(mdoc_file)
    if len(mdoc_files) > 1:
        print(f"WARNING: Multiple .mdoc files found for run name {tiltseries.run.name}, using {base_name}")
    ext = os.path.splitext(base_name)[1]
    dest_file = os.path.join(output_dir, f"{tiltseries.run.id}{ext}")
    s3.get(mdoc_file, dest_file)
    print(f"[{tiltseries.run.id}] Downloaded {base_name} as {os.path.basename(dest_file)}.", flush=True)

    # download tiltseries mrc file
    tiltseries_file = os.path.join(output_dir, f"{tiltseries.run.id}.mrc")
    tiltseries_url = tiltseries.https_mrc_file
    response = requests.get(tiltseries_url, stream=True)
    response.raise_for_status()
    with open(tiltseries_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[{tiltseries.run.id}] Downloaded tiltseries mrc file as {os.path.basename(tiltseries_file)}.", flush=True)

    # create imod file for order list
    mdoc = mdocfile.read(os.path.join(output_dir, f"{tiltseries.run.id}.mdoc"))
    order_list = mdoc['TiltAngle']
    imodpath = os.path.join(output_dir, f"{tiltseries.run.id}_Imod")
    os.makedirs(imodpath, exist_ok=True)
    number = np.arange(len(order_list)) + 1
    
    # save in csv with 'ImageNumber', 'TiltAngle' headers
    df = pd.DataFrame({'ImageNumber': number, 'TiltAngle': order_list})
    df.to_csv(os.path.join(imodpath, f"{tiltseries.run.id}_order_list.csv"), index=False)
