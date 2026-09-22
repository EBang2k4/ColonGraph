# B1 + Node Position Normalization

Scene: `c1_sigmoid2_t4_v4`. Đối chứng: `output/baseline1_nodes2000/c1_sigmoid2_t4_v4`, 30000 fine iterations.

## Cấu hình

- 100 coarse + 30000 fine; 2000 FPS nodes, K=4 inverse-distance, direct translation blending.
- Gaussian KNN centroid loss 0.01; depth 0.25; dcolor 0.005; color smooth 0.0001; TV 0.001, như B1.
- Shared MLP 128 × 3, position PE=10/time PE=4; scale/rotation/color/renderer không sửa.
- `--node_position_normalization bbox`: input MLP là `(xyz - center) / half_extent`, bao gồm raw XYZ và spatial PE. Graph và output displacement ở coordinate gốc.
- `position_center` và `position_scale` (half_extent) lưu trong `deformation_nodes.pth`, fixed suốt fine. Normalization mặc định vẫn `none` trong khi chờ kết quả full.
- Seed code giữ nguyên: `setup_seed(6666)`, sau đó `safe_state` đặt Python/NumPy/Torch về seed 0 trước khi tạo model.

## Kiểm soát khởi tạo cho phép so sánh

Smoke đầu tiên cho thấy chạy lại coarse trên GPU dù cùng seed có sai khác số học: canonical max diff 0.0261, 24 node đổi thứ tự FPS, 81 phần tử association khác sau remap, max weight diff 0.00287. Không dùng graph này cho full comparison.

Runner bbox vì thế dùng `--node_initialization_reference output/baseline1_nodes2000/c1_sigmoid2_t4_v4/point_cloud/iteration_1`.

Quy trình vẫn thực hiện coarse 100; trước fine, phục hồi chính xác trạng thái B1 post-coarse đã lưu **trước optimizer step fine đầu tiên**. Bao gồm canonical Gaussian attributes, appearance HexPlane/color embedding, node centers, associations, weights và MLP khởi tạo. Khôi phục RNG quanh việc load để không đổi chuỗi random do khởi tạo module khi load. Giữ active SH degree và cấu hình training.

Đây không phải teacher warm-start: code từ chối reference nếu tham số MLP khác fresh initialization cùng seed, hoặc config graph/MLP khác. Không dùng checkpoint 7k/30k hoặc teacher-fit MLP. Không freeze canonical; không thêm ARAP hoặc đổi sampling.

Smoke fixed đã xác nhận initial MLP, tất cả Gaussian/appearance attributes, node_xyz, indices và weights **bit-identical** với B1-raw. Normalization center `[-311.8159484863281, -321.0732727050781, -262.2760009765625]`, half_extent `181.6591796875`.

## Kiểm tra và outputs

- `tests/test_sparse_nodes.py`: 9 numerical/CUDA tests, gồm normalization, legacy loading, gradient và renderer roundtrip.
- Smoke corrected: `logs/baseline1_bbox_smoke_fixed/c1_sigmoid2_t4_v4/status.txt`; 100 coarse + 200 fine rồi toàn bộ hậu xử lý. Không lấy smoke metrics để kết luận quality.
- Full: `logs/baseline1_bbox_nodes2000/c1_sigmoid2_t4_v4/status.txt` là trạng thái trực tiếp.
- `bbox_initial_checks.json`: config diff, identical initial state, center/half_extent.
- `results.json`: PSNR, SSIM, LPIPS, Depth MSE.
- `deformation_profile.csv`: CUDA-event duration mỗi 100 fine iterations.
- `b1_bbox_benchmark.json`: B1 raw/bbox checkpoint benchmark trong cùng lần chạy; có deformation_forward_ms (autograd enabled), geometry/MLP/color timing.
- `step_checks.json`: dataset hashes, graph/buffers fixed, MLP/canonical/color có cập nhật, losses/config phù hợp, stages exit 0.
- `b1_bbox_comparison.md/.json`: raw vs normalized đủ metrics và Depth MSE reduction. Không tự đổi default khi chỉ có smoke hoặc chưa review tradeoff RGB/depth.

Runner tự động: train → render test → metrics → look-at → time-shift → benchmark B0 → benchmark B1 → audit → paired bbox audit. Stage lỗi dừng và ghi FAILED.

```bash
bash run_sparse_nodes.sh c1_sigmoid2_t4_v4 30000 baseline1_bbox_nodes2000 2000 0.01 direct bbox
```

Chưa triển khai teacher warm-start, canonical freeze, motion-aware sampling hoặc ARAP. Đây chỉ là các bước tiếp theo có điều kiện dựa trên full metrics.

Smoke đầu `baseline1_bbox_smoke` giữ lại để truy nguyên; không phải run đối chứng hợp lệ (graph drift và lỗi shell do sửa script đang chạy). Không ghi đè hoặc xóa output cũ.

## Bàn giao full run

Smoke fixed hoàn tất toàn bộ stages, exit 0, lúc 2026-09-12 21:14:58 +07. Dataset hash khớp B1; graph/buffers fixed xuyên suốt 200 fine; đầy đủ metrics và paired benchmark được tạo. 9/9 unit/integration tests PASS.

Full run khởi chạy bằng detached session (`start_new_session=True`) ngày 2026-09-12, runner PID `288933`, tag `baseline1_bbox_nodes2000`. Ước lượng 2.5–3 giờ theo các run trước; **chưa có kết quả full tại thời điểm ghi chú này**. Đọc status/log hiện tại thay vì suy ra tiến độ từ tài liệu.
