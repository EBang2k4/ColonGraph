# U7 protocol được đóng gói

U7 transverse nhánh từ F0 actual10k; U7 rectum nhánh từ Stage5 baseline A actual10k. Toàn bộ model/Adam/RNG của prefix được giữ lại; additive và micro head thêm mới khởi tạo bằng0. Tọa độ canonical, geometry, appearance cùng train; không đóng băng nhánh nào.

- Geometry: G3-Surf, GGAS-v2 spatial-only learned K8, KL=.01. Appearance original2000 FPS nodes/K4; node features8D, Gaussian embedding16D.
- RGB: `.80 * mean(sqrt(error²+1e-6)) + .15 * (1-SSIM)/2 + 3.0 * MSE`; TV5e-4; colorvar1e-4.
- LR: native schedules, tất cả nhóm nhân cosine15k→30k, floor0.06. XYZ native final1.6e-6 trước cosine. Prefix resets3k6k9k được giữ; không reset12k.
- Depth: `.25 * native normalized L1 + omega * .30 * mean(sqrt((pred-GT)²+1e-8))/(GTmax+1e-6)`. Predicted max của normalized term có đầy đủ autograd.
- Sensor gate: binary `omega = 1` nếu ratio pooled median TRAIN depth fine0 thuộc[.5,2], ngược lại0. Ratio transverse1.0574552 và rectum0.9143711 nên raw termON ở cả hai. Không phải continuous gate; Stage9 raw coefficient hiệu dụng là.15, U7 tăng lên.30.
- Boundary mask: hợp bottom15% observation counts (`radii>0` qua toàn bộ200TRAIN ở10k) và top15% khoảng cách Euclidean canonical từ centroid. Tie-break theo originalGaussianID. Mask giữ cố định, uniontransverse56674/200000=28.337%, rectum60000/200000=30%.
- Ở10k, toàn bộ opacity logit tăng lên `max(logit10k,logitfine0)`. Sau mỗi Adam, mask boundary bị chặn dưới bằng `min(logitfine0,logit(.5))`. Đây là hard projection theo bất đẳng thức, không thêm một loss mềm. Adam moments cũ được giữ.
- Micro world-space độc lập camera: `alpha * .002 * a * tanh(head)`, alpha ramp10k→12k, bound theo từng coordinate; `a` là NodeMLP frozen bbox half-extent. Micro bằng0 trên toàn bộ mask. Rawmicro L2=.001. Không chiếu ray camera.
- KNN99=.01: `mean(||x_t - mean99(x_t) - gamma_i * ref10k||²)` với canonical reference offset10k detached. Interiorgamma.75 ramp10k→11k; boundarygamma0. Đây không phải neo XYZ vào vị trí cố định.

Masks/opacity sampling là geometric proxies, không chứng nhận topology giải phẫu. Opacity floor làm thay đổi sampling CH/HD95; một thay đổi metric không tự chứng minh bề mặt giải phẫu tốt hơn.

Geometry evaluation: exact scipyKDTree; CH=sum hai directional mean squaredNN; HD95=max hai directional95-percentile EuclideanNN. Scene units, không ICP/scale fitting/downsampling. Native opacity-dependent1…10 Gaussian samples, fixed seed1729+frame, renderer-activated scale clamp.05*spatial_lr_scale. Hai chiều được báo riêng.

Checkpoint loader khôi phục `stage11.reference`, `stage11.boundary`, `stage11.floor`, gate và micro_iteration cùng native model/Adam. Tensor state giữ nguyên; path metadata được thay bằng token `@bundle/` và `@dataset/` để relocation. Không đổi weights, architecture hoặc loss của baseline gốc.
