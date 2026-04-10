@echo off
REM V-JEPA 2 Setup — Windows + RTX 5070 Ti
REM Run in Anaconda Prompt (not regular cmd)

echo === V-JEPA 2 Windows Setup ===
echo Target: RTX 5070 Ti (Blackwell, CUDA 12.8)
echo.

REM ── 1. Conda env ──────────────────────────────────────────────────────────
call conda create -n vjepa python=3.11 -y
call conda activate vjepa

REM ── 2. PyTorch + CUDA 12.8 ────────────────────────────────────────────────
REM Blackwell (SM_100) needs PyTorch 2.7+.
REM If stable fails for your card, switch to the nightly line below.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
REM pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

REM ── 3. V-JEPA 2 core deps ─────────────────────────────────────────────────
pip install timm einops

REM ── 4. Server + client deps ───────────────────────────────────────────────
pip install fastapi "uvicorn[standard]" python-multipart opencv-python numpy pillow requests huggingface_hub safetensors

REM ── 4b. DepthAI (OAK-D cameras) ──────────────────────────────────────────
REM Installs depthai + depthai-nodes (same as on the Pis)
pip install depthai depthai-nodes

REM ── 5. Clone V-JEPA 2 repo ────────────────────────────────────────────────
cd /d %USERPROFILE%
if not exist vjepa2 (
    git clone https://github.com/facebookresearch/vjepa2.git
) else (
    echo vjepa2 repo already present, skipping clone
)

REM ── 6. Download ViT-L weights (~1.2 GB) ───────────────────────────────────
echo.
echo Downloading V-JEPA 2 ViT-L weights from HuggingFace...
python -c "from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id='facebook/vjepa2-vitl-fpc64-256', local_dir=os.path.join(os.environ['USERPROFILE'], 'vjepa2-weights', 'vitl'), ignore_patterns=['*.md','*.txt']); print('Weights downloaded.')"

echo.
echo === Setup complete ===
echo.
echo Quick-start:
echo   Terminal 1:  conda activate vjepa  and  python server.py
echo   Terminal 2:  conda activate vjepa  and  python oak_client.py --name usb-oak --display
echo   Terminal 3:  conda activate vjepa  and  python oak_client.py --ip ^<your-camera-ip^> --name eth-oak --display
echo.
echo Find your Ethernet OAK-D IP:
echo   python -c "import depthai as dai; [print(d.deviceId, d.name) for d in dai.Device.getAllAvailableDevices()]"
echo.
echo Health check once server is up:
echo   curl http://localhost:8765/health
pause
