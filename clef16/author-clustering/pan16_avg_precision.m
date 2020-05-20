function AP=pan16_avg_precision(T,A)

% It calculates Average Precision for ranking of document pairs in A
% according to ground truth A

if numel(A)==0
    AP=0;
    return;
end

AI=[];
AS=[];
AF=cell(numel(A),2);
for i=1:numel(A)
    AF{i,1}=A{i}.document1;
    AF{i,2}=A{i}.document2;
    AI=[AI;i];
    AS=[AS;A{i}.score];
end

% In case of multiple entries for the same document pair, it keeps the
% first one
if numel(A)>1
    I=[];
    for i=numel(A):-1:2
        for j=i-1:-1:1
            if (strcmp(AF{i,1},AF{j,1})==1 && strcmp(AF{i,2},AF{j,2})==1) || (strcmp(AF{i,1},AF{j,2})==1 && strcmp(AF{i,2},AF{j,1})==1)
                I=[I;i];
                break;
            end
        end
    end
%    I
    AF(I,:)=[];
    AI(I)=[];
    AS(I)=[];
end

% It deletes pairs that contain the same document twice
if numel(A)>1
    I=[];
    for i=1:size(AF,1)
        if strcmp(AF{i,1},AF{i,2})==1
            I=[I;i];
        end
    end
%    I
    AF(I,:)=[];
    AI(I)=[];
    AS(I)=[];
end

TF=cell(numel(T),2);
for i=1:numel(T)
    TF{i,1}=T{i}.document1;
    TF{i,2}=T{i}.document2;
end

[~,S]=sort(AS,'descend');
AP=0;
Correct=0;
for i=1:numel(S)
    I1=find(strcmp(TF(:,1),AF{S(i),1})==1);
    I2=find(strcmp(TF(:,2),AF{S(i),2})==1);
    I3=find(strcmp(TF(:,2),AF{S(i),1})==1);
    I4=find(strcmp(TF(:,1),AF{S(i),2})==1);
    if sum(ismember(I1,I2))>0 || sum(ismember(I3,I4))>0
        Correct=Correct+1;
        AP=AP+Correct/i;
%        disp(Correct/i)
    end
end

AP=AP/size(TF,1);
