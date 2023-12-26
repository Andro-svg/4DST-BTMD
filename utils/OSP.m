function [Criteria,tp] = OSP(d,U,X,no_lines,no_rows,groundtruth)
% 正交子空间投影检测器
% input：
% d：L*1，先验光谱信息，感兴趣的目标
% U：不感兴趣的目标
% X: L*N，检验的高光谱图像
% output：
% matrix：N*1，检测结果

% 创建时间；20221121 创建者：时艺丹
% matrix = [];

tic;

L = size(U,1);
Pup = eye(L,L) - (U*pinv(U'*U)*U');
% Pup = (U*pinv(U'*U)*U');
% w_osp = X' * Pup * d * pinv(d'*Pup*d);
w_osp = d' * Pup * X;

tp=reshape(w_osp,[no_lines,no_rows]);      % matrix

% target_map=255*(tp-min(min(tp))*ones(size(tp,1),size(tp,2)))/(max(max(tp))-min(min(tp)));
% temp=reshape(target_map,no_lines*no_rows,1);         % vector

% matrix=[matrix temp];               % vector
% figure(),colormap('gray');
% imagesc(target_map);
% figure(),colormap('gray');
% imagesc(abs(tp));
% subplot(3,5,tt); colormap; imagesc(abs(tp));
figure();colormap; imagesc(abs(tp));
tp_criticize = reshape(tp,[no_lines*no_rows,1]);
[Criteria,~,~]=Cal_3DROC(tp_criticize,groundtruth);

toc;

end

