# Fonctions de plusieurs variables
#load("fig-calculdiff-03.sage")

reset()

var('x,y')
f = x^2+y*sin(x+y^2)

from sage.plot.plot3d.plot3d import axes

G = plot3d(f,(x,-1,1),(y,-1,1), adaptive=True, color='orange') #, max_bend=.1, max_depth=15)
G = G + axes(2.5,1, color='black')

#GG = G.rotate((0,1,0),pi/8)
#GG = GG.rotate((0,0,1),pi/10)

G.show(aspect_ratio=1,frame=False)
#G.save('fig-calculdiff-03.png')


