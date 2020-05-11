% PAN-2016 evaluation scrpipt for the author clustering task
% for Octave 3.8.2

if nargin~=6
    disp('Usage: -i DATASET-DIR -a ANSWERS-DIR -o OUTPUT-DIR')
    return;
end

PARAMS=['-i';'-a';'-o'];

for i=1:2:nargin
    for j=1:size(PARAMS,1)
        if strcmp(lower(argv(){i}),'-a')==1 && strcmp(PARAMS(j,:),'-a')==1
            Run=argv(){i+1};
            PARAMS(j,:)='  ';
        end
        if strcmp(lower(argv(){i}),'-i')==1 && strcmp(PARAMS(j,:),'-i')==1
            Dataset=argv(){i+1};
            PARAMS(j,:)='  ';
        end
        if strcmp(lower(argv(){i}),'-o')==1 && strcmp(PARAMS(j,:),'-o')==1
            Output=argv(){i+1};
            PARAMS(j,:)='  ';
        end
    end
end

if size(find(PARAMS(:,1)=='-'),1)>0
    disp('Usage: -i DATASET-DIR -a ANSWERS-DIR -o OUTPUT-DIR')
    return;
end

X=fileread([Dataset,filesep,'info.json']);
INFO=parseJSON(X);

S=strcat('[',10);
MAP=[];
MF=[];
for i=1:numel(INFO)
    LANGUAGE=INFO{i}.language;
    GENRE=INFO{i}.genre;
    FOLDER=INFO{i}.folder;
    if isdir([Dataset,filesep,'truth',filesep,FOLDER])==0
        disp(['Error: Ground truth folder ',FOLDER,' not found']);
        return
    end
    X=fileread([Dataset,filesep,'truth',filesep,FOLDER,filesep,'clustering.json']);
    TRUE_CLUSTERING=parseJSON(X);
    X=fileread([Dataset,filesep,'truth',filesep,FOLDER,filesep,'ranking.json']);
    TRUE_RANKING=parseJSON(X);
    if isdir([Run,filesep,FOLDER])==0
        RUN_CLUSTERING=[];
        RUN_RANKING=[];
    else
        X=fileread([Run,filesep,FOLDER,filesep,'clustering.json']);
        RUN_CLUSTERING=parseJSON(X);
        X=fileread([Run,filesep,FOLDER,filesep,'ranking.json']);
        RUN_RANKING=parseJSON(X);
    end
    [R,P,F]=pan16_bcubed(TRUE_CLUSTERING,RUN_CLUSTERING);
    AP=pan16_avg_precision(TRUE_RANKING,RUN_RANKING);
    MAP=[MAP,AP];
    MF=[MF,F];
    disp(['problem: ',FOLDER,' language: ',LANGUAGE,' genre: ',GENRE,' F-Bcubed: ',num2str(F),' R-Bcubed: ',num2str(R),' "P-Bcubed: ',num2str(P),' Av-Precision: ',num2str(AP)])
    S=strcat(S,' {"problem": "',FOLDER,'",',10,'  "language": "',LANGUAGE,'",',10,'  "genre": "',GENRE,'",',10,'  "F-Bcubed": ',num2str(F),',',10,'  "R-Bcubed": ',num2str(R),',',10,'  "P-Bcubed": ',num2str(P),',',10,'  "Av-Precision": ',num2str(AP),'},',10,10);
end

if S(end-2)==','
    S(end-2)=[];
    S=[S ']'];
end

fid=fopen([Output,filesep,'out.json'],'w');
fprintf(fid,'%s',S);
fclose(fid);

SP=sprintf('measure {\n key: "Mean F-score"\n value: "%f"\n}\nmeasure {\n key: "Mean Average Precision"\n value: "%f"\n}\n',mean(MF),mean(MAP));
fid=fopen([Output,filesep,'evaluation.prototext'],'w');
fprintf(fid,'%s',SP);
fclose(fid);
