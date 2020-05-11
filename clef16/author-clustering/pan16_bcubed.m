function [B3Recall,B3Precision,B3Fscore]=pan16_bcubed(T,A)

% It calculates Bcubed Recall, Precision, and F-score for clustering in A
% according to ground truth T

B3Precision=0;
B3Recall=0;
B3Fscore=0;

if numel(A)==0
    B3Recall=0;
    B3Precision=0;
    B3Fscore=0;
    return;
end

k=0;
AI=[];
for i=1:numel(A)
    for j=1:numel(A{i})
        k=k+1;
        AF{k}=A{i}{j}.document;
        AI=[AI;i];
    end
end

if numel(unique(AF))~=numel(AF)
    disp('Error: overlapping clusters')
    return;
end

k=0;
TI=[];
for i=1:numel(T)
    for j=1:numel(T{i})
        k=k+1;
        TF{k}=T{i}{j}.document;
        TI=[TI;i];
    end
end

% In case some documents are not included in answers, they considered to
% belong to distinct singleton clusters
K=numel(AF);
C=max(AI);
for i=1:numel(TF)
    I=find(strcmp(AF,TF{i})==1);
    if numel(I)==0
        K=K+1;
        C=C+1;
        AF{K}=TF{i};
        AI=[AI;C];
    end
end

B3P=0;
B3R=0;
for i=1:numel(TF)
    Relevant=find(TI==TI(i));
    I=find(strcmp(AF,TF{i})==1);
    if numel(I)==1
        Retrieved=find(AI==AI(I(1)));
    end
    Correct=0;
    for j=1:numel(Relevant)
        if find(strcmp(AF(Retrieved),TF{Relevant(j)})==1)>0
%            disp(TF{Relevant(j)})
            Correct=Correct+1;
        end
    end
    B3P=B3P+Correct/numel(Retrieved);
    B3R=B3R+Correct/numel(Relevant);
end

B3Precision=B3P/numel(TF);
B3Recall=B3R/numel(TF);
B3Fscore=2*B3Recall*B3Precision/(B3Recall+B3Precision);
