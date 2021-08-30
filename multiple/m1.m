function m1_sym = m1(in1,sigma)
    
    [g_,~] = g(in1);
    m1_sym = f(in1) + sigma*norm(g_,1) + sigma*norm(max(h(in1), 0),1);
end