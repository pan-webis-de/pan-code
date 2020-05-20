% Evaluation script for authorship attribution problems

if nargin~=8
    disp('Usage: -a ANSWERS-FILE -t TRUTH-FILE -m META-FILE -o OUT-FILE')
    return;
end

PARAMS=['-a';'-t';'-m';'-o'];

for i=1:2:nargin
    for j=1:size(PARAMS,1)
        if strcmp(lower(argv(){i}),'-t')==1 && strcmp(PARAMS(j,:),'-t')==1
            GTF=argv(){i+1};
            PARAMS(j,:)='  ';
        end
        if strcmp(lower(argv(){i}),'-a')==1 && strcmp(PARAMS(j,:),'-a')==1
            ANSF=argv(){i+1};
            PARAMS(j,:)='  ';
        end
        if strcmp(lower(argv(){i}),'-m')==1 && strcmp(PARAMS(j,:),'-m')==1
            MFF=argv(){i+1};
            PARAMS(j,:)='  ';
        end
        if strcmp(lower(argv(){i}),'-o')==1 && strcmp(PARAMS(j,:),'-o')==1
            OUTF=argv(){i+1};
            PARAMS(j,:)='  ';
        end
    end
end

if size(find(PARAMS(:,1)=='-'),1)>0
    disp('Usage: -a ANSWERS-FILE -t TRUTH-FILE -m META-FILE -o OUT-FILE')
    return;
end

% Reading the input files
ANST=fileread(ANSF);
ANS=parseJSON(ANST);
GTT=fileread(GTF);
GT=parseJSON(GTT);
MFT=fileread(MFF);
MF=parseJSON(MFT);

A=cell(numel(MF.candidate_authors));
for i=1:numel(MF.candidate_authors)
    A{i}=MF.candidate_authors{i}.author_name;
end

% Building the confusion matrix
CM=zeros(numel(A)+1,numel(A)+1);
for i=1:numel(GT.ground_truth)
    TRUE=size(CM,1);
    PRED=size(CM,2);
    for k=1:numel(A)
        if strcmp(A{k},GT.ground_truth{i}.true_author)==1
            TRUE=k;
            break;
        end
    end
    for j=1:numel(ANS.answers)
        if strcmp(GT.ground_truth{i}.unknown_text,ANS.answers{j}.unknown_text)==1
            for k=1:numel(A)
                if strcmp(A{k},ANS.answers{j}.author)==1
                    PRED=k;
                    break;
                end
            end
            break;
        end
    end
    CM(TRUE,PRED)=CM(TRUE,PRED)+1;
end

if sum(CM(:,end))+sum(CM(end,:))==0
    CM(:,end)=[];
    CM(end,:)=[];
    disp('Closed-set attribution:')
else disp('Open-set attribution:')
end

% Calculating evaluation scores
COR=0;
CORR=[];
for i=1:size(CM,1)
    COR=COR+CM(i,i);
    CORR=[CORR CM(i,i)];
end

Accuracy=COR/sum(sum(CM));
disp(['Accuracy = ',num2str(Accuracy),' (',int2str(COR),'/',int2str(sum(sum(CM))),')'])

RETR=sum(CM,1);
REL=sum(CM,2);
Precision=CORR(RETR~=0)./RETR(RETR~=0);
Recall=[CORR(REL~=0)'./REL(REL~=0)]';
Macro_Precision=mean(Precision);
Macro_Recall=mean(Recall);
Macro_F1=2*Macro_Recall*Macro_Precision/(Macro_Recall+Macro_Precision);
disp(['Macro-F1 = ',num2str(Macro_F1)])
disp(['Macro-Recall = ',num2str(Macro_Recall)])
disp(['Macro-Precision = ',num2str(Macro_Precision)])

PROTOTEXT=['measure{',10,'  key  : "accurracy"',10,'  value: "',num2str(Accuracy),'"',10,'}',10,...
           'measure{',10,'  key  : "macro-f1"',10,'  value: "',num2str(Macro_F1),'"',10,'}',10,...
           'measure{',10,'  key  : "macro-recall"',10,'  value: "',num2str(Macro_Recall),'"',10,'}',10,...
           'measure{',10,'  key  : "macro-precision"',10,'  value: "',num2str(Macro_Precision),'"',10,'}'];

% Saving the results in the output file
fid=fopen([OUTF],'w','n');
fprintf(fid,'%s',PROTOTEXT);
fclose(fid);

