function [x_dot,fx,fu] = F(t,x,u)
    x_dot = dynamics(t,x,u);
    fx = dFds(t,x,u);
    fu = dFdu(t,x,u);
end