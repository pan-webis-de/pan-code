function P=extract_profile(TEXT,n,L)

% PROFILE Extracts the most frequent units (words or n-grams) from text
%   TEXT : The input text
%      n : 0 for words >0 for n-grams (n-gram length)
%      L : Profile size
%      P : Profile in the form [Frequency Units]

if n==0
    TEXT1=lower(TEXT);
    [U,S,Len]=words(TEXT1);
else [U,S,Len]=ngrams(n,TEXT);
end

A=[S/Len U];
A=sortrows(A);
if size(A,1)>=L
    P=A(end-L+1:end,:);
else
    P=A;
end
