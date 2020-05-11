function data = parseJSON(string)
    % DATA = parseJSON(string)
    % This function parses a JSON string and returns a cell array with the
    % parsed data. JSON objects are converted to structures and JSON arrays are
    % converted to cell arrays.

    % lassytouton, 2011
    %
    % Octave/MATLAB JSON parser (compatible with Octave 3.2.4 and MATLAB R2011b)
    %
    % This parser is basically François Glineur's parser from the MATLAB File Exchange,
    % http://www.mathworks.com/matlabcentral/fileexchange/23393-another-json-parser,
    % with minor modifications to make it compatible with Octave

    global jsonString;

    global pos;
    global len;

    global esc;

    global index_esc;
    global len_esc;

    jsonString = string;

    pos = 1;
    len = length(jsonString);
    % String delimiters and escape characters are identified beforehand to improve speed
    esc = regexp(jsonString, '["\\]'); index_esc = 1; len_esc = length(esc);

    if pos <= len
        switch(next_char)
            case '{'
                data = parse_object;
            case '['
                data = parse_array;
            otherwise
                error_pos('Outer level structure must be an object or an array');
        end
    end
end

function object = parse_object
    parse_char('{');
    object = [];
    if next_char ~= '}'
        while 1
            str = parse_string;
            if isempty(str)
                error_pos('Name of value at position %d cannot be empty');
            end
            parse_char(':');
            val = parse_value;
            object.(valid_field(str)) = val;
            if next_char == '}'
                break;
            end
            parse_char(',');
        end
    end
    parse_char('}');
end

function object = parse_array
    parse_char('[');
    object = cell(0, 1);
    if next_char ~= ']'
        while 1
            val = parse_value;
            object{end+1} = val;
            if next_char == ']'
                break;
            end
            parse_char(',');
        end
    end
    parse_char(']');
end

function parse_char(c)
    global jsonString;

    global pos;
    global len;

    skip_whitespace;
    if pos > len || jsonString(pos) ~= c
        error_pos(sprintf('Expected %c at position %%d', c));
    else
        pos = pos + 1;
        skip_whitespace;
    end
end

function c = next_char
    global jsonString;

    global pos;
    global len;

    skip_whitespace;
    if pos > len
        c = [];
    else
        c = jsonString(pos);
    end        
end
    
function skip_whitespace
    global jsonString;

    global pos;
    global len;

    while pos <= len && isspace(jsonString(pos))
        pos = pos + 1;
    end
end

function str = parse_string
    global jsonString;

    global pos;
    global len;

    global esc;

    global index_esc;
    global len_esc;

    if jsonString(pos) ~= '"'
        error_pos('String starting with " expected at position %d');
    else
        pos = pos + 1;
    end
    str = '';
    while pos <= len
        while index_esc <= len_esc && esc(index_esc) < pos 
            index_esc = index_esc + 1;
        end
        if index_esc > len_esc
            str = [str jsonString(pos:end)];
            pos = len + 1;
            break;
        else
            str = [str jsonString(pos:esc(index_esc)-1)];
            pos = esc(index_esc);
        end
        switch jsonString(pos)
            case '"' 
                pos = pos + 1;
                return;
            case '\'
                if pos+1 > len
                    error_pos('End of file reached right after escape character');
                end
                pos = pos + 1;
                switch jsonString(pos)
                    case {'"' '\' '/'}
                        str(end+1) = jsonString(pos);
                        pos = pos + 1;
                    case {'b' 'f' 'n' 'r' 't'}
                        str(end+1) = sprintf(['\' jsonString(pos)]);
                        pos = pos + 1;
                    case 'u'
                        if pos+4 > len
                            error_pos('End of file reached in escaped unicode character');
                        end
                        str(end+1:end+6) = jsonString(pos-1:pos+4);
                        pos = pos + 5;
                end
            otherwise % should never happen
                str(end+1) = jsonString(pos);
                pos = pos + 1;
        end
    end
    error_pos('End of file while expecting end of string');
end

function num = parse_number
    global jsonString;

    global pos;
    global len;

    num = [];

    [start, finish] = regexp(jsonString(pos:len), '^[\w]?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?[\w]?', 'once');
%    [start, finish] = regexp(jsonString(pos:min(len,pos+20)), '^[\w]?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?[\w]?', 'once');
    if ~isempty(start)
        [num, one] = sscanf(jsonString(pos + start - 1:pos + finish - 1), '%f', 1); % TODO : compare with json(pos:end);

        pos = pos + finish;
    else
        error_pos('Error reading number at position %d');
    end
end

function val = parse_value
    global jsonString;

    global pos;
    global len;

    switch(jsonString(pos))
        case '"'
            val = parse_string;
            return;
        case '['
            val = parse_array;
            return;
        case '{'
            val = parse_object;
            return;
        case {'-','0','1','2','3','4','5','6','7','8','9'}
            val = parse_number;
            return;
        case 't'
            if pos+3 <= len && strcmpi(jsonString(pos:pos+3), 'true')
                val = true;
                pos = pos + 4;
                return;
            end
        case 'f'
            if pos+4 <= len && strcmpi(jsonString(pos:pos+4), 'false')
                val = false;
                pos = pos + 5;
                return;
            end
        case 'n'
            if pos+3 <= len && strcmpi(jsonString(pos:pos+3), 'null')
                val = [];
                pos = pos + 4;
                return;
            end
    end
    error_pos('Value expected at position %d');
end

function error_pos(msg)
    global jsonString;

    global pos;
    global len;

    poss = max(min([pos-15 pos-1 pos pos+20],len),1);
    if poss(3) == poss(2)
        poss(3:4) = poss(2)+[0 -1];         % display nothing after
    end

    msg = [sprintf(msg, pos) ' : ... ' jsonString(poss(1):poss(2)) '<error>' jsonString(poss(3):poss(4)) ' ... '];
    error(msg);
end

function str = valid_field(str)   
% From MATLAB doc: field names must begin with a letter, which may be
% followed by any combination of letters, digits, and underscores.
% Invalid characters will be converted to underscores, and the prefix
% "alpha_" will be added if first character is not a letter.
    if ~isletter(str(1))
        str = ['alpha_' str];
    end
    str(~isletter(str) & ~('0' <= str & str <= '9')) = '_';   
end