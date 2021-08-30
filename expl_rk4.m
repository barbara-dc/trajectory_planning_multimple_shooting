function [s,A,B] = expl_rk4(t0,s,u,A,B,h,n_int)
    
    for i=0:(n_int-1)
        ti = t0+h*i;
        [k1,~,~] = F(ti,s,u);
        [k2,~,~] = F(ti + 1/2 * h, s + 1/2 * h * k1, u);
        [k3,~,~] = F(ti + 1/2 * h, s + 1/2 * h * k2, u);
        [k4,~,~] = F(ti + h, s + h * k3, u);
        s = s + 1/6 * h * (k1 +2*k2 + 2*k3 + k4);
        
        A=dsds(ti,s,u)*A;
        B=dsds(ti,s,u)*B+dsdu(ti,s,u);
    end
end