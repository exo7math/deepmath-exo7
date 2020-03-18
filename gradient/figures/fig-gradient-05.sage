# Ligne de gradient
reset()

var('x,y,z')
f = x^3-y^2-x
G = plot3d(f-z,(x,-2,2),(y,-2,2),(z,-2,2))
G.show()

#G.save('fig-gradient-05.png')
