function result = binary_entropy(p)
    if 0 < p && p < 1
        result = -p * log2(p) - (1 - p) * log2(1 - p);
    elseif p == 0 || p == 1
        result = 0;
    else
        error("Input to binary entropy out of range")
    end
end

function H = entropy(P)
    sizeP = numel(P);
    rP = reshape(P, 1, sizeP);
    if abs(sum(rP)-1)>1e-6
        error("Sum of input for entropy not equal to 1");
    end
    H = 0;
    for i = 1:sizeP
        if (rP(i) < 0) || (rP(i) > 1)
            error("Entropy input out of bound");
        elseif (rP(i) == 0) || (rP(i) == 1)
            continue;
        else
            H = H - rP(i) * log2(rP(i));
        end
    end
end

function c = convolve(p,q)
    c = p*(1-q) + q*(1-p);
end


function r = R_GV(delta, w)
    Pxxtc = [1-w-delta/2, delta/2;
             delta/2, w-delta/2];
    Hxxtc = entropy(Pxxtc);
    r = 2 * binary_entropy(w) - Hxxtc;
end


function gv = GV_bound(delta)
    gv = 1 - binary_entropy(delta);
end


delta = 0.3;
jr = (1-sqrt(1-2*delta))/2;
loops=20;
var_w = linspace(jr, 0.5, loops);

ls_rate = [];
ls_gv = [];

for i = 1:loops
    w = var_w(i);
    % rate = Rc+Rs
    rate = 1 - binary_entropy(convolve(delta, w)) + R_GV(delta, w);
    gv_bound = GV_bound(delta);
    ls_rate(end+1) = rate;
    ls_gv(end+1) = gv_bound;
end


plot(var_w, ls_rate, var_w,ls_gv);

