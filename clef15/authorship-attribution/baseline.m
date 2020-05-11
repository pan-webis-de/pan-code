% A baseline approach for authorship attribution
% Stamatatos E. (2007) Author Identification Using Imbalanced and Limited Training Texts
% In Proc. of the 4th International Workshop on Text-based Information Retrieval
% Implementation with fixed values for parameters n and L
% Usage: -i INPUT-FOLDER -o OUT-FILE

n=3;    % order of character n-grams
L=5000; % size of profile

if nargin~=4
    disp('Usage: -i INPUT-FOLDER -o OUT-FILE')
    return;
end

PARAMS=['-i';'-o'];

for i=1:2:nargin
    for j=1:size(PARAMS,1)
        if strcmp(lower(argv(){i}),'-i')==1 && strcmp(PARAMS(j,:),'-i')==1
            DIR=argv(){i+1};
            PARAMS(j,:)='  ';
        end
        if strcmp(lower(argv(){i}),'-o')==1 && strcmp(PARAMS(j,:),'-o')==1
            OUTF=argv(){i+1};
            PARAMS(j,:)='  ';
        end
    end
end

if size(find(PARAMS(:,1)=='-'),1)>0
    disp('Usage: -i INPUT-FOLDER -o OUT-FILE')
    return;
end

% Extracting profiles of candidate authors
X=fileread([DIR,filesep,'meta-file.json']);
J=parseJSON(X);

for i=1:numel(J.candidate_authors)
    D=dir([DIR,filesep,J.candidate_authors{i}.author_name,filesep,'known*.txt']);
    disp([J.candidate_authors{i}.author_name,': ',int2str(numel(D))])
    TEXT=[];
    for j=1:numel(D)
        TEXT=[TEXT fileread([DIR,filesep,J.candidate_authors{i}.author_name,filesep,D(j).name])];
    end
    T(i).profile=extract_profile(TEXT,n,L);
    T(i).text=TEXT;
end

% Attributing unknown files to candidate authors
A=['{',10,'"answers": ['];
DT=dir([DIR,filesep,J.folder,filesep,'unknown*.txt']);
for i=1:numel(DT)
    TEXT=fileread([DIR,filesep,J.folder,filesep,DT(i).name]);
    P1=extract_profile(TEXT,n,L);
    DIST=[];
    for j=1:numel(T)
        P2=T(j).profile;
        [Common Index]=ismember(P1(:,2:end),P2(:,2:end),'rows');
        CC=find(Common);
        D=sum(((2*(P1(CC)-P2(Index(Index>0))))./(P1(CC)+P2(Index(Index>0)))).^2);
        D=(D+4*(size(Common,1)-sum(Common)))/(4*size(P1,1));
        DIST=[DIST,D];
    end
    [M,I]=min(DIST);
    A=[A,10,'{"unknown_text": "',DT(i).name,'","author": "',J.candidate_authors{I}.author_name,'","score": ',num2str(1-DIST(I)),'},'];
    disp([DT(i).name,' ',J.candidate_authors{I}.author_name,' ',num2str(1-DIST(I))]);
end

if A(end)==','
    A(end)=[];
end
A=[A,']',10,'}'];

% Saving the answers in the output file
fid=fopen(OUTF,'w');
fprintf(fid,'%s',A);
fclose(fid);
