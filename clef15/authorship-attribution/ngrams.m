function [NG1,SC,Length]=ngrams(n,FILE)
% Extracts n-grams from a text file
% [NG1,SC,Length] = ngams(n,FILE)
%      n : n-gram length
%   FILE : input text file
%    NG1 : ngram list (in ascii form)
%     SC : score for each n-gram
% Length : amount of total n-grams (used for normalization)

NG=[];
if size(FILE,2)==0
    NG1=[];
    SC=[];
    Length=0;
    return
end

s=size(FILE,2);
for i=n-1:-1:0
    NG=[NG FILE(1,n-i:s-i)'];
end

NG=sortrows(NG);

NNG=double(NG); % Conversion of strings to ascii form
%NNG=[];
%NNG(:,:)=NG(:,:);

[NG1,I,J]=unique(NNG,'rows');
SC=I(2:end,1)-I(1:end-1,1);
SC=[I(1,1);SC];

Length=size(NG,1);