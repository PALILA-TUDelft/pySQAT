% Function for Excel

function export_struct_to_excel(OUT, filename)
    if nargin < 2
        filename = 'loudness_output.xlsx';
    end

    fields = fieldnames(OUT);

    for i = 1:numel(fields)
        field = fields{i};
        value = OUT.(field);

        try
            sheet = field(1:min(31,end)); % Excel sheet name limit
            if isscalar(value)
                writematrix(value, filename, 'Sheet', sheet);
            elseif isvector(value)
                writematrix(value(:), filename, 'Sheet', sheet); % as column
            elseif ismatrix(value)
                writematrix(value, filename, 'Sheet', sheet);
            else
                warning('Skipping field "%s": unsupported data type.', field);
            end
        catch ME
            warning('Failed to write field "%s": %s', field, ME.message);
        end
    end

    fprintf('Export complete: %s\n', filename);
end

% --- Common parameters ---
fs = 48000;
duration = 1.0;
t = linspace(0, duration, fs * duration);
amp = 0.02;
signal = amp * sin(2 * pi * 1000 * t);
field = 0;
time_skip = 0.2;
show = false;

% Export all fields for Method 2
OUT2 = Loudness_ISO532_1(signal', fs, field, 2, time_skip, show);
export_struct_to_excel(OUT2, 'val_method2_MAT.xlsx');
disp('Exported all Method 2 results to a single Excel file.');

% Repeat for Method 1
OUT1 = Loudness_ISO532_1(signal', fs, field, 1, time_skip, show);
export_struct_to_excel(OUT1, 'val_method1_MAT.xlsx');
disp('Exported all Method 1 results to a single Excel file.');

% Repeat for Method 0
spl = zeros(1, 28);
spl(16) = 40;
OUT0 = Loudness_ISO532_1(spl, 1, field, 0, 0, show);
export_struct_to_excel(OUT0, 'val_method0_MAT.xlsx');
disp('Exported all Method 0 results to a single Excel file.');