function [ grid, intersegments ] = line_pixel_intersection( x_grid, y_grid, start_pt, end_pt )
%LINE_PIXEL_INTERSECTION Calculate Intersection of Line on Grid Pixels
%   x_grid, y_grid -- Struct Specifying X,Y-Coordinate of Grid  
%                     start: start position
%                     spacing: position spacing on grid
%                     N: number of points on grid
%   start_pt       -- Start Point of Line [x, y]
%   end_pt         -- End Point of Line [x, y]
%   grid           -- Struct With (x, y) Grid On Which Intersegments Calculated 
%   intersegments  -- Struct Specifying Intersection of Line with Pixels
%                     fragEndPts: (x, y) coords of intersegment endpoints
%                     lengths: matrix of line-pixel intersection lengths in
%                     (i, j, val) format where (i, j) is the row and column
%                     in matrix and val is the length over the pixel.                     

% Create Grid for Pixel Center
grid.x = x_grid.start + (0:x_grid.N-1) * x_grid.spacing; 
grid.y = y_grid.start + (0:y_grid.N-1) * y_grid.spacing; 

% Create Grid for Pixel Vertices
xv = x_grid.start - x_grid.spacing/2 + (0:x_grid.N) * x_grid.spacing; 
yv = y_grid.start - y_grid.spacing/2 + (0:y_grid.N) * y_grid.spacing; 

% Find Line Pixel Intersections
xrng = [start_pt(1), end_pt(1)];
yrng = [start_pt(2), end_pt(2)];
if xrng(1) ~= xrng(2)
    line_param_x = (xv-xrng(1))/(xrng(2)-xrng(1));
else
    line_param_x = [];
end
if yrng(1) ~= yrng(2)
    line_param_y = (yv-yrng(1))/(yrng(2)-yrng(1));
else
    line_param_y = [];
end
line_param = sort([line_param_x, line_param_y]);
line_param = [0, line_param(line_param>=0 & line_param<=1), 1];
xint = xrng(1) + line_param * (xrng(2)-xrng(1));
yint = yrng(1) + line_param * (yrng(2)-yrng(1));

% Sparse Matrix of Intersection Lengths
intersegments.lengths.row = ...
    round(((yint(1:end-1)+yint(2:end))/2-y_grid.start)/y_grid.spacing)+1;
intersegments.lengths.col = ...
    round(((xint(1:end-1)+xint(2:end))/2-x_grid.start)/x_grid.spacing)+1;
intersegments.lengths.val = sqrt(diff(xint).^2 + diff(yint).^2);
intersegments.fragEndPts = [xint(:), yint(:)];

end

