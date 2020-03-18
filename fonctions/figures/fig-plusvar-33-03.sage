# Fonctions de plusieurs variables
#load("fig-calculdiff-03.sage")

reset()

x,y = var('x,y')
t = var('t')

f = x*y/(x^2+y^2)

from sage.plot.plot3d.plot3d import axes

G = plot3d(f,(x,-1,1),(y,-1,1), adaptive=True, plot_points=(200,200), color='orange') #, max_bend=.1, max_depth=15)
G = G + axes(2.5,1, color='black')

G = G + parametric_plot3d( (t,t,1/2), (t,-1,1), color = 'blue',thickness=5) 
G = G + parametric_plot3d( (t,0,0), (t,-1,1), color = 'red',thickness=5) 

G.show(aspect_ratio=1,frame=False)


#GG = G.rotate((0,1,0),pi/8)
#GG = GG.rotate((0,0,1),pi/10)


#G.save('fig-calculdiff-03.png')


