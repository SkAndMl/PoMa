# NSCC Server Setup Guide

This README provides a step-by-step guide for transferring files, setting up the environment, and executing jobs on the NSCC server (`aspire2antu.nscc.sg`).

---

## 📁 File Transfer to NSCC

### 1. Transfer a Single File

```bash
scp path/to/file.py <username>@aspire2antu.nscc.sg:/home/users/<username>/desired/location
```

2. Transfer a Folder

On NSCC (remote): Create the target folder:
```bash
mkdir -p /home/users/<username>/desired/location/folder
```
On Local (your machine): Transfer folder recursively:
```bash
scp -r path/to/folder <username>@aspire2antu.nscc.sg:/home/users/<username>/desired/location
```


⸻

⚙️ Environment Setup on NSCC

Run the following commands once logged into the NSCC server:
```bash
module unload gcc/12.1.0-nscc
module load gcc/8.1.0
module load cuda/11.8.0
module load python/3.10.9
```
Create and activate a Python virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```


⸻

📜 PBS Job Submission Script

Create a shell script (e.g., run_job.sh) with the following contents:

```bash
#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1:mem=48gb
#PBS -l walltime=48:00:00
#PBS -P personal-<user_id>
#PBS -N <any_name>

# Commands start here
cd ${PBS_O_WORKDIR}
module unload gcc/12.1.0-nscc
module load gcc/8.1.0
module load cuda/11.8.0
module load python/3.10.9
source venv/bin/activate

python3.10 train.py
```
Replace <user_id> and <any_name> accordingly.

⸻

🚀 Job Submission

To submit the job script to the NSCC job scheduler, run:

```bash
qsub run_job.sh
```

Notes:
1. scp gpt-k (after you are in the desired branch)
2. scp llama 1B model
3. change llama_path in config.py to the location of the folder in NSCC