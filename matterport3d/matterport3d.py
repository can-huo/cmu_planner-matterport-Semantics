#!/usr/bin/env python3
import argparse
import os
import tempfile
import urllib.request

BASE_URL = "http://kaldir.vc.cit.tum.de/matterport/"
RELEASE = "v1/scans"
RELEASE_TASKS = "v1/tasks/"
TOS_URL = BASE_URL + "MP_TOS.pdf"
FILETYPES = [
    "cameras", "matterport_camera_intrinsics", "matterport_camera_poses",
    "matterport_color_images", "matterport_depth_images", "matterport_hdr_images",
    "matterport_mesh", "matterport_skybox_images", "undistorted_camera_parameters",
    "undistorted_color_images", "undistorted_depth_images", "undistorted_normal_images",
    "house_segmentations", "region_segmentations", "image_overlap_data",
    "poisson_meshes", "sens",
]
TASK_FILES = {
    "habitat": ["mp3d_habitat.zip"],
}

def get_release_scans(release_file):
    with urllib.request.urlopen(release_file) as f:
        return [line.decode("utf-8").strip() for line in f if line.strip()]

def download_file(url, out_file):
    if os.path.isfile(out_file):
        print("skip existing", out_file)
        return
    print(url, ">", out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fh, tmp = tempfile.mkstemp(dir=os.path.dirname(out_file))
    os.close(fh)
    urllib.request.urlretrieve(url, tmp)
    os.rename(tmp, out_file)

def download_scan(scan_id, out_dir, file_types):
    os.makedirs(out_dir, exist_ok=True)
    for ft in file_types:
        url = f"{BASE_URL}{RELEASE}/{scan_id}/{ft}.zip"
        download_file(url, os.path.join(out_dir, ft + ".zip"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out_dir", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--type", nargs="+", required=True)
    args = p.parse_args()

    print("Agree TOS:", TOS_URL)
    input("Press Enter to continue, or Ctrl-C to exit. ")

    scans = get_release_scans(BASE_URL + RELEASE + ".txt")
    if args.id not in scans:
        raise SystemExit("invalid scan id: " + args.id)
    bad = set(args.type) - set(FILETYPES)
    if bad:
        raise SystemExit("invalid type: " + str(bad))

    out_dir = os.path.join(args.out_dir, RELEASE, args.id)
    download_scan(args.id, out_dir, args.type)
    print("done:", out_dir)

if __name__ == "__main__":
    main()
