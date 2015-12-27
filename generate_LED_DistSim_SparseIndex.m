% compute a sparse index for kernel learnig,this is similar with compute a
% graph with k_w and k_b
function G = generate_LED_DistSim_SparseIndex(trn_X,trn_y,k_w,k_b,full)
    nPoints = length(trn_y);
    if nargin<5
        full=false;% set full as false
    end
    if full
        G=ones(nPoints);
        return;
    else
        fprintf('sparese kernel index used!\n');
        Log_trn_X=zeros(size(trn_X));
        for tempk=1:size(trn_X,3)
            Log_trn_X(:,:,tempk)=logm(trn_X(:,:,tempk));
        end
        tmpSim = Compute_LERM_Sim(Log_trn_X,Log_trn_X,true);
        %Within Graph
        G_w = zeros(nPoints);
        for tmpC1 = 1:nPoints
            tmpIndex = find(trn_y == trn_y(tmpC1));
            [~,sortInx] = sort((-tmpSim(tmpIndex,tmpC1))); % descend
            if (length(tmpIndex) < k_w)
                max_w = length(tmpIndex);
            else
                max_w = k_w;
            end
            G_w(tmpC1,tmpIndex(sortInx(1:max_w))) = 1;
        end
        %Between Graph
        G_b = zeros(nPoints);
        for tmpC1 = 1:nPoints
            tmpIndex = find(trn_y ~= trn_y(tmpC1));
            [~,sortInx] = sort((-tmpSim(tmpIndex,tmpC1)));
            if (length(tmpIndex) < k_b)
                max_b = length(tmpIndex);
            else
                max_b = k_b;
            end
            G_b(tmpC1,tmpIndex(sortInx(1:max_b))) = 1;
        end
        G = ((G_w + G_b)+(G_w + G_b)')/2; % pay attention to this
        index=(G~=0); G(index)=1;
        fprintf('In Graph G, %d pair are used!\n',sum(sum(G))/2);
    end
end