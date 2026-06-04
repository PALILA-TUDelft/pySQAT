function acquire(variable, name)
%ACQUIRE  Export a vector to Excel, splitting real/imag parts if needed.
%
%   acquire(variable, name)
%
%   * variable : numeric vector (row or column, real or complex)
%   * name     : char array or string scalar used as the base file-name
%
%   Behaviour
%   ----------
%   ▸ If VARIABLE has more than one element
%     • It checks three sample points (first, last, and ⅓-length) for
%       complex values.
%     • If any of those samples are complex, it writes an Excel workbook
%       <name>_complex.xlsx with two columns:
%           "<name> (Real)"   – real(variable(:))
%           "<name> (Imag)"   – imag(variable(:))
%     • Otherwise it writes <name>.xlsx with a single column:
%           "<name>"          – variable(:)
%   ▸ If VARIABLE has ≤ 1 element nothing is written.

    arguments
        variable {mustBeNumeric}
        name (1,:) char
    end

    if numel(variable) <= 1
        warning('Variable has 1 or 0 elements – nothing exported.'); %#ok<*WNTAG>
        return
    end

    %--- 1. Detect whether the vector should be treated as complex ----------
    len          = numel(variable);
    idxSample    = [1, len, floor(len/3)+1];   % +1 for MATLAB's 1-based indexing
    isComplexSig = ~isreal(variable(idxSample));

    %--- 2. Assemble a table ------------------------------------------------
    if any(isComplexSig)          % at least one sample is complex
        T = table( ...
            real(variable(:)), ...
            imag(variable(:)), ...
            'VariableNames', {sprintf('%s (Real)', name), ...
                              sprintf('%s (Imag)', name)});
        fileName = sprintf('%s_complex.xlsx', name);
    else                           % completely real signal
        T = table(variable(:), 'VariableNames', {name});
        fileName = sprintf('%s.xlsx', name);
    end

    %--- 3. Write to Excel --------------------------------------------------
    writetable(T, fileName, 'FileType', 'spreadsheet');
end
