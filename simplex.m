% Step 1: 读取.mat文件数据
mat_file = 'D:\PSSUNUNET\OURS_samson.mat';  % 替换成实际的mat文件路径
data = load(mat_file);

% 假设矩阵存储在 'A' 键下
abundance_matrix = data.A;  % 替换为你的数据字段名称

% Step 2: 转换为双精度（double）
abundance_matrix = double(abundance_matrix);  % 将丰度矩阵转换为双精度

% Step 3: 获取点集并转置
% 假设数据是 (3, h, w)，每个点对应于 (3, h, w) 的矩阵
[h, w] = size(abundance_matrix, 2, 3);  % 获取 h 和 w 的尺寸
points = reshape(abundance_matrix, 3, h * w);  % 重塑矩阵为 3xN 的形式

% Step 4: 筛选和为1的点
point_sums = sum(points, 1);  % 对每列（每个点）求和
valid_points = points(:, abs(point_sums - 1) < 1e-6);  % 筛选和接近1的点

% Step 5: 计算有效点集的凸包
k = convhull(valid_points');  % k 是凸包面索引，记得转置 `valid_points` 作为 `convhull` 输入

% Step 6: 可视化
figure;

% 绘制有效点
scatter3(valid_points(1,:), valid_points(2,:), valid_points(3,:), 'filled');
hold on;

% 绘制凸包的面
%trisurf(k, valid_points(1,:), valid_points(2,:), valid_points(3,:), 'FaceColor', 'r', 'FaceAlpha', 0.3);

% 绘制由 (1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0) 组成的面
X = [1.0, 0, 0];  % 点1
Y = [0, 1.0, 0];  % 点2
Z = [0, 0, 1.0];  % 点3

% 使用 patch 函数绘制面
patch([X(1), Y(1), Z(1)], [X(2), Y(2), Z(2)], [X(3), Y(3), Z(3)], 'g', 'FaceAlpha', 0.5);

% 设置图形标签
xlabel('X');
ylabel('Y');
zlabel('Z');

% 显示网格
grid on;
axis equal;

% 添加图例
legend('abu', 'Convex Hull Faces', 'Triangle Face');

hold off;
