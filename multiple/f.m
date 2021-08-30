function cost = f(input)
    global s_init
    [R, ~] = multiple_shooting(input, s_init);
    cost = 0.5*(R.'*R);
end