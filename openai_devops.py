import os
import sys
import subprocess
import openai
import time
import platform

# --- CONFIG ----
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

REQUIRED = [
    "torch",
    "torchvision",
    "mmcv",
    "mmpose",
    "opencv-python",
    "matplotlib",
    "tqdm",
    "numpy",
    "pillow",
    "scipy",
    "onnxruntime",
    "xtcocotools",
    "mmengine",
]
PROJECT_ROOT = os.path.abspath(os.getcwd())

POSE_EXTRACTION_CMD = [
    sys.executable, "pose_extraction/extract_pose_from_videos.py",
    "--input_dir", "dataset/CSL_News/rgb_format",
    "--output_dir", "dataset/CSL_News/pose_format",
    "--config", "mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py",
    "--checkpoint", "checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
    "--device", "cuda:0"
]

def ask_openai(question, system="You are a helpful Linux/ML/Python DevOps engineer. Be brief."):
    try:
        chat = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question}
            ],
            max_tokens=700,
            temperature=0.2,
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI API failed:", e)
        return None

def pip_show(package):
    print(f"\nChecking {package}...")
    try:
        pkg_name = package.split('==')[0]
        result = subprocess.run([sys.executable, "-m", "pip", "show", pkg_name],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bool(result.stdout)
    except Exception:
        return False

def try_mmcv_mim_install(torch_ver, cuda_ver):
    # Use openmim to install correct MMCV for torch/cuda
    print(f"Trying MMCV install with openmim (torch={torch_ver}, cuda={cuda_ver})...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "mmcv", "mmcv-full"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "openmim"])
    # Map CUDA version string to wheel URLs
    if cuda_ver:
        cu_major = "".join(cuda_ver.split('.')[:2])  # e.g. "11.8" -> "118"
        url = f"https://download.openmmlab.com/mmcv/dist/cu{cu_major}/torch{torch_ver}/index.html"
    else:
        url = ""
    mmcv_cmd = f"mim install mmcv-full==1.7.0"
    if url:
        mmcv_cmd += f" -f {url}"
    print(f"Running: {mmcv_cmd}")
    result = subprocess.run(mmcv_cmd, shell=True)
    return result.returncode == 0

def try_install_with_ai(package, fail_reason=None, max_attempts=10):
    attempt = 0
    tried_cmds = []
    orig_package = package
    # If MMCV is the package, try the openmim logic first before using AI
    if "mmcv" in package:
        try:
            import torch
            torch_ver = torch.__version__
            cuda_ver = torch.version.cuda
        except Exception:
            torch_ver = None
            cuda_ver = None
        if try_mmcv_mim_install(torch_ver or "1.13.0", cuda_ver or "11.8"):
            return True
    while attempt < max_attempts:
        attempt += 1
        print(f"\n[Attempt {attempt}] Installing {package}...")
        if attempt == 1 and not tried_cmds:
            cmd = f"{sys.executable} -m pip install {package}"
        else:
            if not tried_cmds:
                print("No AI suggestion available, cannot continue.")
                break
            cmd = tried_cmds[-1]
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        if result.returncode == 0:
            print(f"✅ {package} installed successfully!")
            return True
        print(result.stderr)
        ai_question = (
            f"My attempt to install {package} failed with this error:\n{result.stderr if result.stderr else fail_reason}\n"
            f"My Python: {sys.version}\n"
            f"Platform: {platform.platform()}\n"
            f"CUDA: {os.environ.get('CUDA_HOME', 'unknown')}\n"
            f"nvcc: {subprocess.getoutput('nvcc --version')}\n"
            "If the version does not work or is deprecated, suggest a working version or compatible alternative. "
            "What exact pip/conda/shell command should I try next to install it? Only give the command, nothing else."
        )
        ai_cmd = ask_openai(ai_question)
        if not ai_cmd:
            print("OpenAI did not return a suggestion. Manual intervention required.")
            break
        print("OpenAI suggests:", ai_cmd)
        tried_cmds.append(ai_cmd)
        # Try to parse out a new version string for the log
        ai_cmd_pkg = None
        if "pip install" in ai_cmd and "==" in ai_cmd:
            parts = ai_cmd.split()
            for idx, token in enumerate(parts):
                if "==" in token:
                    ai_cmd_pkg = token
                elif idx < len(parts)-1 and parts[idx+1].startswith("=="):
                    ai_cmd_pkg = token + parts[idx+1]
        if ai_cmd_pkg:
            package = ai_cmd_pkg
        time.sleep(2)
    print(f"❌ Failed to install {orig_package} after {max_attempts} attempts.")
    return False

def pip_install(package):
    if pip_show(package):
        print(f"Already installed: {package}")
        return True
    print(f"\n--- Installing {package} ---")
    result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print(f"✅ {package} installed successfully.")
        return True
    print(result.stderr)
    # Use AI to suggest any working version or fix
    return try_install_with_ai(package, fail_reason=result.stderr)

def ensure_dependencies():
    print("\n=== Checking dependencies ===")
    for pkg in REQUIRED:
        if not pip_show(pkg):
            pip_install(pkg)
        else:
            print(f"✔️ Already installed: {pkg}")

def check_cuda():
    try:
        import torch
        print("PyTorch:", torch.__version__)
        if torch.cuda.is_available():
            print("CUDA available:", torch.version.cuda)
        else:
            print("CUDA is NOT available to PyTorch!")
    except Exception as e:
        print("Could not import torch:", e)

def check_ffmpeg():
    ffmpeg = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE)
    if ffmpeg.stdout:
        print(f"ffmpeg found at {ffmpeg.stdout.decode().strip()}")
    else:
        print("ffmpeg NOT found! Install it with: sudo apt-get install ffmpeg")

def auto_fix_and_run_pose_extraction(max_attempts=100):
    print("\n==== TESTING POSE EXTRACTION PIPELINE ====")
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        print(f"\n[Pose Extraction Attempt {attempts}]")
        proc = subprocess.run(POSE_EXTRACTION_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(proc.stdout)
        if proc.returncode == 0:
            print("🎉 Pose extraction completed successfully!")
            return True
        print("❌ Pose extraction failed! Error:\n", proc.stderr)
        with open(POSE_EXTRACTION_CMD[1], "r") as f:
            code = f.read()
        suggestion = ask_openai(
            f"The following pose extraction command failed with this error:\n{' '.join(POSE_EXTRACTION_CMD)}\n{proc.stderr}\n"
            f"Here is the script code:\n{code}\n"
            "If the failure is due to outdated/deprecated packages or API changes, update the code or install commands as needed. "
            "Reply with the corrected code OR the shell command(s) needed to fix it (try a different version if necessary). Only output code or shell commands.",
            "You are a top-tier ML/MLOps engineer. Only output actionable fixes."
        )
        if not suggestion:
            print("No AI fix suggestion. Manual intervention required.")
            break
        # Try to interpret if suggestion is a Python script or a shell command
        if suggestion.strip().startswith(("python", "pip", "conda", "apt-get", "sudo", "bash", "./", "sh ")):
            print("AI fix is a shell command. Executing:", suggestion)
            subprocess.run(suggestion, shell=True)
        else:
            print("AI fix looks like a Python script (or partial). Attempting to replace file and retry.")
            # backup and replace code
            backup = POSE_EXTRACTION_CMD[1] + f".ai_fix_attempt_{attempts}"
            os.rename(POSE_EXTRACTION_CMD[1], backup)
            with open(POSE_EXTRACTION_CMD[1], "w") as f:
                f.write(suggestion)
            print(f"Backed up old script to {backup}. Retrying.")
        time.sleep(2)
    print("❌ Failed to auto-fix pose extraction after multiple attempts.")
    return False

def main():
    print("🔍 Starting OpenAI DevOps Agent")
    print(f"Working directory: {PROJECT_ROOT}")

    ensure_dependencies()
    check_cuda()
    check_ffmpeg()

    print("\n\n--- Environment checked and dependencies installed. ---\n")
    print("Now testing the pose extraction pipeline with:")
    print(" ".join(POSE_EXTRACTION_CMD))
    auto_fix_and_run_pose_extraction()

if __name__ == "__main__":
    main()

