import os
import subprocess
import sys

# Datasets configuration
# You can adjust the patient range or add Siena URL if available
CHB_MIT_PATIENTS = [f"chb{i:02d}" for i in range(15, 19)] # Patients 15 to 18 as current focus
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"

def check_dependencies():
    """Check if wget is installed."""
    res = subprocess.run(["which", "wget"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print("[-] Error: 'wget' is required but not installed.")
        sys.exit(1)

def download_chbmit():
    raw_dir = os.path.expanduser('~/seizure-prediction/data/raw/chbmit')
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"[+] Starting smart download for CHB-MIT patients: {CHB_MIT_PATIENTS}")
    
    for patient in CHB_MIT_PATIENTS:
        patient_url = f"{BASE_URL}{patient}/"
        print(f"\n[==>] Processing {patient} from PhysioNet...")
        
        # Smart wget command:
        # -r: recursive
        # -N: don't re-retrieve files unless newer on server (smart caching)
        # -c: continue broken downloads (resume)
        # -np: don't ascend to parent directory
        # -nH: disable generation of host directories
        # --cut-dirs=3: strip first 3 directory levels from URL structure
        cmd = [
            "wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3",
            patient_url,
            "-P", raw_dir
        ]
        
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"[✓] Successfully downloaded/verified {patient}")
        else:
            print(f"[-] Warning/Error occurred while downloading {patient} (Exit code: {result.returncode})")

def main():
    check_dependencies()
    download_chbmit()
    print("\n[✓] All download tasks completed successfully!")

if __name__ == "__main__":
    main()
