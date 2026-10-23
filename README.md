# ColonGraph_unified — U7 baseline

Đóng gói baseline **Stage 11 U7** cho DynamicColon `data_transverse` và `data_rectum`.
Code, checkpoint, mask và graph reference nằm trong thư mục này. Dataset và Conda environment được dùng riêng.

## Dùng ngay trên máy hiện tại

```bash
cd /home/robot/colongraph/ColonGraph_unified

# Kiểm tra SHA của bản đóng gói (chỉ đọc).
./run.sh --help
python3 scripts/verify_package.py

# Kiểm tra checkpoint U7 30k, mask/floor và model/Adam state.
CUDA_VISIBLE_DEVICES=0 ./run.sh validate --scene transverse
CUDA_VISIBLE_DEVICES=1 ./run.sh validate --scene rectum

# Xuất video reconstruction của TEST200.
CUDA_VISIBLE_DEVICES=0 ./run.sh render --scene transverse
CUDA_VISIBLE_DEVICES=1 ./run.sh render --scene rectum

# Đánh giá TEST200: RGB, depth, FPS và hai chiều CH/HD95.
CUDA_VISIBLE_DEVICES=0 ./run.sh evaluate --scene transverse --geometry
CUDA_VISIBLE_DEVICES=1 ./run.sh evaluate --scene rectum --geometry

# Kiểm tra train 3 updates từ prefix 10k, rồi cold-load model/Adam.
CUDA_VISIBLE_DEVICES=0 ./run.sh smoke --scene transverse --steps 3

# Train lại phần U7: prefix 10k được đóng gói -> fine 30k.
CUDA_VISIBLE_DEVICES=0 ./run.sh train --scene transverse --out output/transverse_retrain
CUDA_VISIBLE_DEVICES=1 ./run.sh train --scene rectum --out output/rectum_retrain
```

`run.sh` kích hoạt Conda `.env` ở thư mục cha. Có thể chỉ định `COLONGRAPH_CONDA_PREFIX` và `COLONGRAPH_CONDA_BASE`; hoặc source `activate.sh` rồi dùng `python u7.py ...`.

Dataset mặc định: `../datasets/DynamicColon/data_{transverse,rectum}`. Dùng `--data /duong/dan/data_transverse` khi chuyển dataset sang vị trí khác. Metadata của scene được kiểm tra trước khi dùng prefix đã đóng gói.

Output mặc định nằm trong `output/`, thư mục mới theo thời gian. `--out` phải là thư mục chưa tồn tại. Checkpoint mới nằm trong `<out>/model/`. Ví dụ tiếp tục checkpoint U7 đã lưu ở 15k:

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh train --scene transverse \
  --run output/transverse_retrain/model --iteration 15000 --end 30000 \
  --out output/transverse_resume
```

## Phạm vi chính xác

- Hai scene đã thực nghiệm: transverse và rectum, mỗi scene 200.000 Gaussian, G3-Surf + learned spatial J8, appearance K4 cố định ở 2.000 FPS nodes.
- **Prefix 10k có sẵn**, gồm model, Adam, RNG, G3/J8/KNN99. U7 thêm 20k updates đến 30k. Lệnh `train` tái thực hiện phần U7, không dựng lại coarse/prefix từ đầu.
- Prefix bắt nguồn từ RGB/depth coarse100; GT PLY không dùng để khởi tạo hoặc training. Checkpoint cuối 30k, mask/floor và canonical reference đều được kèm theo.
- TRAIN: toàn bộ 200 ảnh root. TEST: toàn bộ 200 ảnh trong folder `test/`.
- R6 + Additive025 + world-space MicroDisp195; các thông số cụ thể trong [docs/U7_PROTOCOL.md](docs/U7_PROTOCOL.md).
- Dataset mới cần coarse/prefix, scale gate và mask TRAIN mới; không được ghép dataset khác với prefix/mask hiện tại.
- Bản này đóng gói **U7 DynamicColon**. P/sigmoid2 không thuộc baseline này.

`evaluate --geometry` đọc `ply/pc_0001.ply`…`pc_0200.ply` chỉ ở bước đánh giá sau training và tạo cache trong output. Bỏ `--geometry` để chỉ đo RGB/depth/FPS. `--max-frames 2` là kiểm tra một phần, được ghi rõ `partial: true`; không thay bảng kết quả TEST200.

## Kết quả gốc tại fine30k

| Scene | PSNR | SSIM | LPIPS-A | Raw depth MSE | Normalized depth MSE | CH | HD95 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transverse | 34.416100 | 0.953586 | 0.208249 | 0.02411265 | 0.0003143100 | 0.07057610 | 0.61610891 | 110.78 |
| rectum | 33.732378 | 0.960449 | 0.171300 | 0.20798758 | 0.0004006378 | 0.26466275 | 1.04774167 | 119.72 |

Đây là kết quả gốc của một lần train mỗi nhánh. Các ngưỡng đồng thời Stage 11 **chưa đạt**. Thời gian train thêm20k là 33.30 / 33.06 phút; không bao gồm prefix, pilot và evaluation. FPS phụ thuộc phiên đo và GPU.

Chi tiết: [reference_results/U7_metrics.json](reference_results/U7_metrics.json), metric từng frame và báo cáo Stage 11 trong `reference_results/`.

## Bố cục

```text
u7.py, run.sh, activate.sh       entrypoint chính
assets/{transverse,rectum}/
  prefix_10000/                 checkpoint prefix + optimizer/RNG
  u7_30000/                    checkpoint U7 đã train
  boundary/                    mask, floor, opacity fine0, visibility counts
  references/                  KNN99, G3 reference, K8 candidates
  scale_gate.json              số đo TRAIN fine0, không chọn bằng TEST
configs/U7_*.json               cấu hình checkpoint chốt
scene/, utils/, train.py        native model/renderer/loss
tools/u7_engine.py             U7 training và checkpoint lifecycle
reference_results/             kết quả TEST200 gốc
verification/                  bằng chứng kiểm tra bản đóng gói
scripts/verify_package.py       xác minh manifest
submodules/                    source CUDA dependencies + licenses
```

Các helper trong `tools/` giữ tên lịch sử để bảo toàn native implementation; dùng `u7.py` làm entrypoint. Loader này khôi phục đầy đủ mask/floor/reference của U7, không dùng một loader Gaussian chung thiếu các trường này.

## Môi trường

Đã kiểm tra bằng Conda Python3.9 / Torch2.6.0+cu126 / RTX A4000. `environment.lock.yml` là bản export môi trường đã dùng; `requirements.txt` ghi các phiên bản Python package chính. Việc tạo môi trường trên máy mới chưa được kiểm thử trong lần đóng gói này. CUDA toolkit/compiler và extension cần tương thích với Torch/GPU.

Source extension nằm ở `submodules/simple-knn`, `submodules/diff-gaussian-rasterization-radegs` và `submodules/diff-gaussian-rasterization-texture`. Các binary đã cài trong `.env` không được chép vào folder; trên máy mới cần build/cài chúng trong Conda tương ứng. Giữ nguyên LICENSE và license của các submodule.
