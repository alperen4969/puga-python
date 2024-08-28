function newpop = selectOp(~, pop)
% TOURNAMENT with pref
% Function: newpop = selectOp(opt, pop)
% Description: Selection operator, use binary tournament selection.
%
%         LSSSSWC, NWPU
%    Revision: 1.1  Data: 2011-07-12
%*************************************************************************

popsize = length(pop);
pool = zeros(1, popsize);   % pool : the individual index selected
randnum = randi(popsize, [1, 2 * popsize]);
j = 1;
for i = 1:2:(2*popsize)
    p1 = randnum(i);
    p2 = randnum(i+1);
    % Crowded-comparison operator (NSGA-II)
    result = crowdingComp( pop(p1), pop(p2) );
    if(result == 1)
        pool(j) = p1;
    else
        pool(j) = p2;
    end
    j = j + 1;
end
newpop = pop( pool );

function result = crowdingComp( guy1, guy2)
% Function: result = crowdingComp( guy1, guy2)
%   1 = guy1 is better than guy2
%   0 = other cases
%*************************************************************************
if (guy1.PUI < guy2.PUI)  % better ranking solution is selected
    result = 1;
else
    result = 0;
end





