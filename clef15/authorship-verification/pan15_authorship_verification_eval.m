% PAN-2015 evaluation scrpipt for the author identification task
% for Octave 3.8.2

if nargin~=6
    disp('Usage: -i INPUT-DIR -t TRUTH-FILE -o OUTPUT-FILE')
    return;
end

PARAMS=['-i';'-t';'-o'];
CODES={'DU','Dutch','Essays';'EN','English','Novels';'GR','Greek','Articles';'SP','Spanish','Articles';'EE','English','Essays';'DE','Dutch','Essays';'DR','Dutch','Reviews'};
CODES2={'DU','nl','essays';'EN','en','novels';'GR','gr','articles';'SP','es','articles';'EE','en','essays';'DE','nl','essays';'DR','nl','reviews'};

for i=1:2:nargin
    for j=1:size(PARAMS,1)
        if strcmp(lower(argv(){i}),'-t')==1 && strcmp(PARAMS(j,:),'-t')==1
            TRUTH=argv(){i+1};
            PARAMS(j,:)='  ';
        end
        if strcmp(lower(argv(){i}),'-i')==1 && strcmp(PARAMS(j,:),'-i')==1
            IN=argv(){i+1};
            PARAMS(j,:)='  ';
        end
        if strcmp(lower(argv(){i}),'-o')==1 && strcmp(PARAMS(j,:),'-o')==1
            OUT=argv(){i+1};
            PARAMS(j,:)='  ';
        end
    end
end
disp(PARAMS)
if size(find(PARAMS(:,1)=='-'),1)>0
    disp('Usage: -i INPUT-DIR -t TRUTH-FILE -o OUTPUT-FILE')
    return;
end

% Calculation of ROC-AUC and c@1 scores for the Author Identification task @PAN-2014
% TRUTH: The ground truth file (see format at pan.webis.de)
% ANSWERS: The answers file of a given submission

AUC=0;
C1=0;

% Reading ground truth and answers files
C=dir(TRUTH);
if size(C,1)>0
    GTR=fileread(TRUTH);
    GTR(GTR==13)=10;
else disp(['The given ground truth file (',TRUTH,') does not exist'])
    return;
end
C=dir(IN);
if size(C,1)>0
    ANS=fileread(IN);
    ANS(ANS==13)=10;
else disp(['The given input file (',IN,') does not exist'])
    return;
end

IYes=strfind(ANS,' Y');
INo=strfind(ANS,' N');
if size(IYes,2)+size(INo,2)==0

% Extracting Problem IDs, True answers, and Given answers
% All unanswered problems are assigned the value 0.5
PROBLEMS=[];
TA_ALL=cell(size(CODES,1)+1,1);
GA_ALL=cell(size(CODES,1)+1,1);
for i=1:size(CODES,1)
        TA=[];
        GA=[];
        PATTERN=CODES{i,:};
        I=strfind(GTR,PATTERN);
        for j=1:size(I,2)
            S=textscan(GTR(I(j):end),'%s',1);
            ProblemID=S{1}{1};
            S=textscan(GTR(I(j)+size(ProblemID,2):end),'%s',1);
            TrueAnswer=S{1}{1};
            PROBLEMS=[PROBLEMS;ProblemID];
            if TrueAnswer=='Y'
                TA=[TA;1];
            else TA=[TA;0];
            end
            IA=strfind(ANS,ProblemID);
            if size(IA,2)>0
                S=textscan(ANS(IA(1)+size(ProblemID,2):end),'%s',1);
                GivenAnswer=S{1}{1};
            else GivenAnswer='0.5';
            end
            GivenAnswer1=GivenAnswer;
            GivenAnswer=str2num(GivenAnswer);
            if size(GivenAnswer,1)==0
                GivenAnswer=0.5;
            end
            if GivenAnswer1=='Y'
                GivenAnswer=1;
            end
            if GivenAnswer1=='N';
                GivenAnswer=0;
            end
            GA=[GA;GivenAnswer];
%            disp([ProblemID,' ',TrueAnswer,' ',num2str(GivenAnswer)])
        end
        GA_ALL{i,1}=GA;
        TA_ALL{i,1}=TA;
        GA_ALL{end,1}=[GA_ALL{end,1};GA];
        TA_ALL{end,1}=[TA_ALL{end,1};TA];
end

RESULTS=zeros(size(CODES,1),2);
for i=1:size(TA_ALL,1)
    if size(TA_ALL{i,1},1)>0
        % Calculation of ROC-AUC
        [~,~,AUC]=pan15_authorship_verification_eval_compute_roc(GA_ALL{i,1},TA_ALL{i,1});

        % Calculation of c@1
        B=GA_ALL{i,1};
        B(GA_ALL{i,1}>0.5)=1;
        B(GA_ALL{i,1}<0.5)=0;
        Nc=sum(TA_ALL{i,1}==B);
        N=size(TA_ALL{i,1},1);
        Nu=sum(B==0.5);
        C1=(1/N)*(Nc+(Nu*Nc/N));
        RESULTS(i,1)=AUC;
        RESULTS(i,2)=C1;
    end
end

% Displaying results and saving in the output directory
C=0;
for i=1:size(CODES2,1)
    if size(TA_ALL{i,1})>0
        C=C+1;
        X=['{"language": "',CODES2{i,2},'",',10,'"genre": "',CODES2{i,3},'",',10,'"AUC": ',num2str(RESULTS(i,1)),',',10,'"C1": ',num2str(RESULTS(i,2)),',',10,'"finalScore": ',num2str(RESULTS(i,1)*RESULTS(i,2)),'},',10];
        fprintf(['{\n"genre": "',CODES2{i,3},'",\n"language": "',CODES2{i,2},'",\n"AUC": "', num2str(RESULTS(i,1)),'",\n"C@1": "',num2str(RESULTS(i,2)),'",\n"AUC*C@1": "',num2str(RESULTS(i,1)*RESULTS(i,2)), '"\n}']);
        PROTOTEXT=['measure{',10,'  key  : "language"',10,'  value: "',CODES2{i,2},'"',10,'}',10,...
        'measure{',10,'  key  : "genre"',10,'  value: "',CODES2{i,3},'"',10,'}',10,...
        'measure{',10,'  key  : "AUC"',10,'  value: "',num2str(RESULTS(i,1)),'"',10,'}',10,...
        'measure{',10,'  key  : "C1"',10,'  value: "',num2str(RESULTS(i,2)),'"',10,'}',10,...
        'measure{',10,'  key  : "finalScore"',10,'  value: "',num2str(RESULTS(i,1)*RESULTS(i,2)),'"',10,'}'];
%    else disp([CODES2{i,1},': No ground truth problems found'])
    end
end

if C>1
    X=['{"language": "all",',10,'"genre": "all",',10,'"AUC": ',num2str(RESULTS(end,1)),',',10,'"C1": ',num2str(RESULTS(end,2)),',',10,'"finalScore": ',num2str(RESULTS(end,1)*RESULTS(end,2)),'}',10,'  ]',10,'}'];
    PROTOTEXT=['measure{',10,'  key  : "language"',10,'  value: "all"',10,'}',10,...
        'measure{',10,'  key  : "genre"',10,'  value: "all"',10,'}',10,...
        'measure{',10,'  key  : "AUC"',10,'  value: "',num2str(RESULTS(end,1)),'"',10,'}',10,...
        'measure{',10,'  key  : "C1"',10,'  value: "',num2str(RESULTS(end,2)),'"',10,'}',10,...
        'measure{',10,'  key  : "finalScore"',10,'  value: "',num2str(RESULTS(end,1)*RESULTS(end,2)),'"',10,'}'];
    disp(['ALL: AUC=',num2str(RESULTS(end,1)),' C@1=',num2str(RESULTS(end,2)),' AUC*C@1=',num2str(RESULTS(end,1)*RESULTS(end,2))])
else X=[X(1:end-2)];
end

fid=fopen(OUT,'w');
fprintf(fid,'%s',X);
fclose(fid);

% save results to prototext file
prototextfile=fopen(strrep (OUT, '.txt', '.prototext'),'w');
fprintf(prototextfile,'%s',PROTOTEXT);
fclose(prototextfile);

end

if size(IYes,2)+size(INo,2)>0

% FOR PAN-2013 output
% Extracting Problem IDs, True answers, and Given answers
% All unanswered problems are assigned the value 0.5
PROBLEMS=[];
TA_ALL=cell(size(CODES,1)+1,1);
GA_ALL=cell(size(CODES,1)+1,1);
GBA_ALL=cell(size(CODES,1)+1,1);
GANS=textscan(ANS,'%s');
for i=1:size(CODES,1)
        TA=[];
        GA=[];
        GBA=[];
        PATTERN=CODES{i,:};
        I=strfind(GTR,PATTERN);
        for j=1:size(I,2)
            S=textscan(GTR(I(j):end),'%s',1);
            ProblemID=S{1}{1};
            S=textscan(GTR(I(j)+size(ProblemID,2):end),'%s',1);
            TrueAnswer=S{1}{1};
            PROBLEMS=[PROBLEMS;ProblemID];
            if TrueAnswer=='Y'
                TA=[TA;1];
            else TA=[TA;0];
            end
            IA=find(ismember(GANS{1},ProblemID));
            if size(IA,1)>0
                S=GANS{1}{IA+1};
                if S=='Y'
                    GivenBINAnswer=1;
                end
                if S=='N'
                    GivenBINAnswer=0;
                end
                if S~='Y' && S~='N'
                    GivenBINAnswer=0.5;
                end
                if size(GANS{1},1)>IA+1
                    S=str2num(GANS{1}{IA+2});
                    if size(S,1)==0
                        GivenAnswer=0.5;
                    else GivenAnswer=S;
                    end
                end
            else GivenAnswer=0.5;
                 GivenBINAnswer=0.5;
            end
            GA=[GA;GivenAnswer];
            GBA=[GBA;GivenBINAnswer];
%            disp([ProblemID,' ',TrueAnswer,' ',num2str(GivenBINAnswer),' ',num2str(GivenAnswer)])
        end
        GA_ALL{i,1}=GA;
        GBA_ALL{i,1}=GBA;
        TA_ALL{i,1}=TA;
        GA_ALL{end,1}=[GA_ALL{end,1};GA];
        GBA_ALL{end,1}=[GBA_ALL{end,1};GBA];
        TA_ALL{end,1}=[TA_ALL{end,1};TA];
end

RESULTS=zeros(size(CODES,1),2);
for i=1:size(TA_ALL,1)
    if size(TA_ALL{i,1},1)>0
        % Calculation of ROC-AUC
        [~,~,AUC]=pan15_authorship_verification_eval_compute_roc(GA_ALL{i,1},TA_ALL{i,1});

        % Calculation of c@1
        Nc=sum(TA_ALL{i,1}==GBA_ALL{i,1});
        N=size(TA_ALL{i,1},1);
        Nu=sum(GBA_ALL{i,1}==0.5);
        C1=(1/N)*(Nc+(Nu*Nc/N));
        RESULTS(i,1)=AUC;
        RESULTS(i,2)=C1;
    end
end

% Displaying results and saving in the output directory
C=0;
for i=1:size(CODES2,1)
    if size(TA_ALL{i,1})>0
        C=C+1;
        X=['{"language": "',CODES2{i,2},'",',10,'"genre": "',CODES2{i,3},'",',10,'"AUC": ',num2str(RESULTS(i,1)),',',10,'"C1": ',num2str(RESULTS(i,2)),',',10,'"finalScore": ',num2str(RESULTS(i,1)*RESULTS(i,2)),'},',10];
        PROTOTEXT=['measure{',10,'  key  : "language"',10,'  value: "',CODES2{i,2},'"',10,'}',10,...
        'measure{',10,'  key  : "genre"',10,'  value: "',CODES2{i,3},'"',10,'}',10,...
        'measure{',10,'  key  : "AUC"',10,'  value: "',num2str(RESULTS(i,1)),'"',10,'}',10,...
        'measure{',10,'  key  : "C1"',10,'  value: "',num2str(RESULTS(i,2)),'"',10,'}',10,...
        'measure{',10,'  key  : "finalScore"',10,'  value: "',num2str(RESULTS(i,1)*RESULTS(i,2)),'"',10,'}'];
        fprintf(['{\n"genre": "',CODES2{i,3},'",\n"language": "',CODES2{i,2},'",\n"AUC": "', num2str(RESULTS(i,1)),'",\n"C@1": "',num2str(RESULTS(i,2)),'",\n"AUC*C@1": "',num2str(RESULTS(i,1)*RESULTS(i,2)), '"\n}']);
%    else disp([CODES2{i,1},': No ground truth problems found'])
    end
end

if C>1
    X=['{"language": "all",',10,'"genre": "all",',10,'"AUC": ',num2str(RESULTS(end,1)),',',10,'"C1": ',num2str(RESULTS(end,2)),',',10,'"finalScore": ',num2str(RESULTS(end,1)*RESULTS(end,2)),'}',10];
    PROTOTEXT=['measure{',10,'  key  : "language"',10,'  value: "all"',10,'}',10,...
        'measure{',10,'  key  : "genre"',10,'  value: "all"',10,'}',10,...
        'measure{',10,'  key  : "AUC"',10,'  value: "',num2str(RESULTS(end,1)),'"',10,'}',10,...
        'measure{',10,'  key  : "C1"',10,'  value: "',num2str(RESULTS(end,2)),'"',10,'}',10,...
        'measure{',10,'  key  : "finalScore"',10,'  value: "',num2str(RESULTS(end,1)*RESULTS(end,2)),'"',10,'}'];
    disp(['ALL: AUC=',num2str(RESULTS(end,1)),' C@1=',num2str(RESULTS(end,2)),' AUC*C@1=',num2str(RESULTS(end,1)*RESULTS(end,2))])
else X=[X(1:end-2)];
end

fid=fopen(OUT,'w');
fprintf(fid,'%s',X);
fclose(fid);

% save results to prototext file
prototextfile=fopen(strrep (OUT, '.txt', '.prototext'),'w');
fprintf(prototextfile,'%s',PROTOTEXT);
fclose(prototextfile);

end
