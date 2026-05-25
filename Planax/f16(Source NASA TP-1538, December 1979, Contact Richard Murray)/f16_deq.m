%  function xd = f16_deq(u,x,c);
%
%
%  Usage: xd = f16_deq(u,x,c);
%
%  To compile and link: fmex f16_deq.f f16_deqg.F
%
%  Description:
%
%    Computes the state derivative vector for the F-16 model
%    based on NASA TP-1538, December 1979.
%
%  input:
%    
%    u = input vector = [ thtl  (0 <= thtl <= 1.0)
%                          el   (deg)
%                          ail  (deg)
%                          rdr  (deg)
%			   vxturb (ft/sec),
%			   vyturb (ft/sec),
%			   vzturb (ft/sec)];
%
%    x = state vector = [   vt  (ft/sec)
%                         alpha (rad)
%                          beta (rad)
%                           phi (rad)
%                           the (rad)
%                           psi (rad)
%                            p  (rad/sec)
%                            q  (rad/sec)
%                            r  (rad/sec)
%                           xn  (ft)
%                           xe  (ft)
%                            h  (ft)  
%                           pow (percent, 0 <= pow <= 100) ];
%
%    c = vector of constants:  c(1) through c(9) = inertia constants.
%                              c(10) = aircraft mass, slugs.
%                              c(11) = xcg, longitudinal c.g. location,
%                                      distance normalized by the m.a.c.
%
%  output:
%
%    xd = state vector time derivative.
%
%
