function [B,T,Noise] = LRSD(D,tenP, opts)

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'gamma');         gamma = opts.gamma;              end
if isfield(opts, 'lambda1');        lambda1 = opts.lambda1;                end
if isfield(opts, 'lambda2');        lambda2 = opts.lambda2;                end
if isfield(opts, 'alpha');        alpha = opts.alpha;                end
if isfield(opts, 'beta');        beta = opts.beta;                end
if isfield(opts, 'r');        r = opts.r;                end
if isfield(opts, 'r_max');        r_max = opts.r_max;                end
if isfield(opts, 'mu');             mu       = opts.mu;                   end
%% Initialization
Nway = size(D); %[n1,n2,n3,n4]
N = ndims(D);
B = zeros(Nway);
T = zeros(Nway);
Noise = zeros(Nway);
U = cell(N,1);
A = cell(N,1);
for n=1:N 
    U{n} = ones([Nway(n),r(n)]);
    A{n} = ones([Nway(n),r(n)]);
end
myA = cell(N,1);

M1 = ones(Nway);
M2 = ones(Nway);
M = cell(N,1);
for n=1:N
    M{n} = zeros([Nway(n),r(n)]);
end
G = tenzeros(r);
W = ones(size(T))./ (tenP+0.01);
preNumT = numel(T);

for iter = 1 : max_iter
    size(ttm(G,A,1:N));
    %% update Ui
    temp = prod(alpha)*prod(r);
    for n = 1:N
        U{n} = prox_l1(A{n}+M{n},temp*beta(n)/mu);
    end

    %% update Ai
    for n = 1:N
        tmp = ttm(G,A,-n);
        tmp = double(tenmat(tmp,n));
        Bn = double(tenmat(B,n));
        M2n = double(tenmat(M2,n));
        A{n} = ((Bn-M2n)*tmp'+(U{n}-M{n}))/(eye(size(tmp*tmp'))+tmp*tmp');
    end

    %% update G
    for n=1:N
        myA{n}=A{n};
    end
    
    myG = tucker_solve(G, B, myA, M2, r);
    G = tensor(myG);

    %% update T
    tempT = double(D-B-Noise-M1);
    max(max(max(max(double(tempT)))));
    min(min(min(min(double(tempT)))));

    thres = W *(lambda1/mu);
    max(max(max(max(double(thres)))));
    min(min(min(min(double(thres)))));

    T = prox_l1(tempT, thres); 
    max(max(max(max(double(T)))));
    min(min(min(min(double(T)))));

    %% update B
    B = double((D-T-Noise+ttm(G, A, 1:N)+M2-M1));


    %% update W
    W = 1 ./ ( (abs(T))+ 0.01) ./ (tenP+0.01) ;

    %% update Noise
    Noise = mu*(D-B-T-M1)/(2*lambda2+mu);

    
    %% check the convergence
    currNumT = sum(T(:) > 0); 
    chg =norm(D(:)-T(:)-B(:)-Noise(:))/norm(D(:));
%     fprintf('iter = %d   res=%.10f  \n', iter, chg);
    if (chg < tol) || (currNumT == preNumT)
        break;
    end
    preNumT = currNumT;

    %% update Lagrange multipliers Mi and penalty parameter mu
    M1 = M1 - (D-B-T-N);
    M2 = M2 - (B-ttm(G,A,1:N));
    for n = 1:N
        M{n} = M{n} - (U{n} - A{n});
    end
    mu = min(gamma*mu,max_mu); 


    if r(1)<r_max(1)
        G = tensor(padarray(double(G), [1 1 1 1], 0, 'post'));
        for n=1:4
            A{n} = padarray(A{n}, [0 1], 0, 'post');
            U{n} = padarray(U{n}, [0 1], 0, 'post');
            M{n} = padarray(M{n}, [0 1], 0, 'post');
            r(n)=r(n)+1;
        end
    end
    
end
end

function G = tucker_solve(G, B, A, M, r)
    N = numel(size(B));  
    G_vec = reshape(double(G), [], 1);  % Convert G to a vector
    
    for i = 2
        A_i = A{i};
        B_i = double(unfold(B, i));
        M_i = double(unfold(M, i));
        clear A_cut_i;
        
        A_iT_A_i = A_i' * A_i;  % Compute A_i^T * A_i
        A_cut_i = kron(A{4}, A{3});
        A_cut_i = kron(A_cut_i, A{1});
        A_iT_Bi_Ai = kronmult(A, i, B_i, A_cut_i);  % Compute A_i^T * B_{(i)} * A^{(\i)}
        A_iT_Mi_Ai = kronmult(A, i, M_i, A_cut_i);  % Compute A_i^T * M_{(i)} * A^{(\i)}

        % Compute the vectorized form of the iterative formula
        vec_term = vec(A_iT_Bi_Ai - A_iT_Mi_Ai);
        
        tmp = kron(A_cut_i' * A_cut_i, A_iT_A_i);
        G_vec = pinv(tmp) * vec_term;  % A\b for inv(A)*b
    end

    G = reshape(G_vec, r); 
end


function X = unfold(A, mode)
    sizeA = size(A);
    unfolded_size = [sizeA(mode), prod(sizeA) / sizeA(mode)];
    X = reshape(A, unfolded_size);
end


% Helper function for performing Kronecker product and matrix-vector multiplication
function C = kronmult(A, i, B, A_cut_i)
    % Perform Kronecker product
    C = A{i}' * B;
    
    % Perform matrix-vector multiplication
    size(A_cut_i);
    C = C * A_cut_i;
end

% Helper function for stacking the columns of a matrix or tensor into a single column vector
function v = vec(A)
    v = A(:);
end