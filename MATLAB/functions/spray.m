function val = spray(val_mask, mask)
%SPRAY Adjoint of Masking Operation Used to Sample Values
%   val = spray(val_mask, mask)
%   INPUT:
%       val_mask = values at masked locations
%       mask = masked locations in output array [logical array]
%           (numel(val_mask) must equal nnz(mask))
%   OUTPUT:
%       val = full array with values at masked locations and zero elsewhere
    val = zeros(size(mask));
    val(mask) = val_mask;
end

