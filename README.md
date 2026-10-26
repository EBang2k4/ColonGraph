# ColonGraph — M2_nearest16

Source sạch để **train từ đầu**: coarse100 bước + fine30.000 bước.
Không kèm checkpoint, graph reference, node IDs, Gaussian cloud, cache, dataset hay kết quả thí nghiệm.
Mọi trạng thái/model/reference của run mới đều được tạo trong thư mục output do bạn chọn.

## Lấy source từ GitHub

```bash
git clone --branch baseline/m2-nearest16-final https://github.com/EBang2k4/ColonGraph.git ColonGraph_M2_final
cd ColonGraph_M2_final
```

## Cài đặt bằng Conda

Trong workspace hiện tại:

```bash
cd /home/robot/colongraph/ColonGraph_M2_final
source activate.sh
```

Trên máy mới:

```bash
conda env create -f environment.yml
conda activate colongraph_m2
bash scripts/install.sh
```

CUDA extensions, `simple-knn` và GLM headers đã kèm source trong `submodules/`;
clone thông thường là đủ, không cần tải Git submodule. CUDA extensions được build từ source này. Môi trường đã kiểm tra: Python3.9,
PyTorch2.6.0/CUDA12.6, RTX A4000. Không cần trỏ đến thư mục thí nghiệm cũ.
LPIPS dùng pretrained perceptual networks của dependency khi chạy metrics;
chúng không phải checkpoint ColonGraph và không được đóng gói trong source.

## Chạy pipeline đầy đủ

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh \
  /home/robot/colongraph/datasets/DynamicColon/data_transverse \
  output/data_transverse --geometry

CUDA_VISIBLE_DEVICES=1 bash run.sh \
  /home/robot/colongraph/datasets/DynamicColon/data_rectum \
  output/data_rectum --geometry

CUDA_VISIBLE_DEVICES=0 bash run.sh \
  /home/robot/colongraph/ColonSplat_data/data/coloncrafter/c1_sigmoid2_t4_v4 \
  output/sigmoid2
```

`run.sh` lần lượt train → render TEST → metrics → lookatcamera. Output phải là
thư mục mới. `--geometry` chỉ dùng cho DynamicColon để đo CH/HD95 sau training.

Chạy từng bước, giao diện tương tự ColonSplat:

```bash
python train.py -s /path/to/dataset -m output/scene
python render.py -s /path/to/dataset -m output/scene --skip_train --skip_video
python metrics.py -s /path/to/dataset -m output/scene
python render_lookat_cameras.py -s /path/to/dataset -m output/scene
python test_geometry.py -s /path/to/DynamicColon -m output/scene
```

`--iteration` của render/metrics mặc định chọn checkpoint cuối trong **run mới**.
`--max_frames 2` chỉ là probe. CLI train không có tham số input checkpoint/resume.

## Dataset và khởi tạo

DynamicColon cần `transforms.json`, `img/`, `depth/`, cùng `test/transforms.json`,
`test/img/`, `test/depth/`. Toàn bộ ảnh root dùng TRAIN; thư mục `test/` chỉ dùng TEST.
Khởi tạo200.000 Gaussian từ RGB/depth frame TRAIN đầu, không đọc GT `ply/` khi train.
GT PLY chỉ được đọc bởi metrics khi bật `--geometry`. Lookat dùng `lookat_cams/` của dataset.

C3VDv2 giữ reader và split gốc ColonSplat (`color/`, `depth/`, `pose.txt`, 1/8 frame TEST).
`points3d.ply` nếu có là point cloud **đầu vào của dataset**, không phải model checkpoint;
reader dùng như ColonSplat gốc. Nếu thiếu, điểm được dựng lại từ RGB/depth/pose,
không ghi ngược vào dataset. Số Gaussian theo point cloud đầu vào; sigmoid2 chuẩn có364.500.
C3VD lookat là10 camera overview được tạo từ geometry của run mới, không có matched GT.

## Các giai đoạn tự dựng

| Mốc | Thao tác |
|---|---|
| Coarse100 | Tối ưu raw Gaussian từ RGB/depth; tạo Gaussian KNN99 ban đầu |
| Sau coarse | Tạo2.000 FPS appearance K4 và geometry; C3VD thêm400 importance geometry node |
| Fine2k | Dựng mới E4/G3 surface graph, sửa K4 tối thiểu và trajectory gating từ TRAIN |
| Fine5k | Dựng mới spatial-only J8 candidates; học skinning+KL0,01, ramp5k→6k |
| Fine10k | Một lượt TRAIN đóng băng để chọn harmonic node, tối đa150; tạo M2 nearest16 |
| Fine10k→30k | Joint geometry/appearance với M2; harmonic handshake500 bước |

M2 lấy16 cạnh gần nhất **trong99 ứng viên ban đầu**, sắp theo XYZ hiện tại tại10k;
weights isotropic và reference edges được khóa từ chính run này. Loss0,01 gồm ba
field riêng: node+harmonic, effective micro, canonical edge. Giữ công thức gốc:
DynamicColon gamma1 interior/0 peripheral mask; C3VD gamma1; không dùng Stage16 surface kernel.

Geometry và appearance đều học đến30k. Appearance luôn2.000 node K4 cố định,
R6 + Additive025 + MicroDisp ở nhánh cuối. DynamicColon giữ R6 đến10k rồi thêm
hai zero heads; C3VD dùng Additive/Micro từ fine1 theo protocol P. DynamicColon
chỉ áp opacity floor mỗi bước trên transverse; rectum không có step floor.
Hyperparameters cơ sở nằm trong `configs/m2.json`; lịch theo modality nằm trong source.

## Output và kiểm tra

Model được lưu tại `point_cloud/iteration_*` và `fine_state_*.pth` trong output;
`setup/` chứa các graph/reference/selector được tạo trong run mới. Render xuất
RGB, canonical-only RGB, depth PNG/raw NPY, GT, video và camera manifests.
Metrics gồm PSNR, SSIM, LPIPS-A/V, depth MSE, FPS, và CH/HD95 khi có metric GT.
C3VD chỉ báo normalized estimated-prior depth, không có CH/HD95 GT.

```bash
# Kiểm tra nhanh TẤT CẢ chuyển giai đoạn, không phải benchmark30k.
python train.py -s /path/to/dataset -m /tmp/m2_smoke --smoke

# Kiểm tra source không lẫn checkpoint/results/cache.
python scripts/verify_source.py
```

Smoke chạy100 coarse +12 fine updates và nén các mốc lịch để kiểm tra G3/J8/selector/M2.
Bản source đã được kiểm tra smoke trên transverse, rectum, sigmoid2 và cold reload
model/Adam/harmonic. Chưa train full30k bằng bản source mới; không gán metrics của run cũ cho nó.
