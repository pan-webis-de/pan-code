% PAN-2016 random basline scrpipt for the author clustering task
% for Octave 3.8.2

if nargin~=4
    disp('Usage: -i DATASET-DIR -o OUTPUT-DIR')
    return;
end

PARAMS=['-i';'-o'];

for i=1:2:nargin
    for j=1:size(PARAMS,1)
        if strcmp(lower(argv(){i}),'-i')==1 && strcmp(PARAMS(j,:),'-i')==1
            DIR1=argv(){i+1};
            PARAMS(j,:)='  ';
        end
        if strcmp(lower(argv(){i}),'-o')==1 && strcmp(PARAMS(j,:),'-o')==1
            DIR2=argv(){i+1};
            PARAMS(j,:)='  ';
        end
    end
end

if size(find(PARAMS(:,1)=='-'),1)>0
    disp('Usage: -i DATASET-DIR -o OUTPUT-DIR')
    return;
end

XINFO=fileread([DIR1,filesep,'info.json']);
INFO=parseJSON(XINFO);

for i=1:numel(INFO)
%    LANGUAGE=INFO{i}.language;
%    GENRE=INFO{i}.genre;
    FOLDER=INFO{i}.folder;
    DIR=[DIR1,filesep,FOLDER];
    DIROUT=[DIR2,filesep,FOLDER];
    if isdir(DIROUT)==0
        mkdir(DIROUT);
    end
    % Reads document files and randomly assigns them to authors
    D=dir([DIR,filesep,'*.txt']);
    S=numel(D);
    R=randi(S,S);
    A=R(1,:);
    AUTHORS=unique(A);
    % Writes clustering.json output file
    X=sprintf('[\n');
    for i=1:numel(AUTHORS)
        I=find(A==AUTHORS(i));
        X=sprintf('%s\t[\n',X);
        for j=1:numel(I)
            X=sprintf('%s\t\t{"document": "%s"},\n',X,D(I(j)).name);
        end
        if X(end-1)==','
            X(end-1)=[];
        end
        X=sprintf('%s\t],\n',X);
    end
    if X(end-1)==','
        X(end-1)=[];
    end
    X=sprintf('%s\n]',X);
    fid=fopen([DIROUT,filesep,'clustering.json'],'w');
    fprintf(fid,'%s',X);
    fclose(fid);
    % Writes ranking.json output file
    Y=sprintf('[\n');
    for i=1:numel(AUTHORS)
        I=find(A==AUTHORS(i));
        if numel(I)>1
            for j=1:numel(I)-1
                for k=j+1:numel(I)
                    % Randomly chosen score
                    score=rand(1);
                    Y=sprintf('%s\t{"document1": "%s",\n\t"document2": "%s",\n\t"score":  %s},\n',Y,D(I(j)).name,D(I(k)).name,num2str(score));
                end
            end
        end
    end
    if Y(end-1)==','
        Y(end-1)=[];
    end
    Y=sprintf('%s\n]',Y);

    fid=fopen([DIROUT,filesep,'ranking.json'],'w');
    fprintf(fid,'%s',Y);
    fclose(fid);
end