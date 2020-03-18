# Ligne de gradient
#load("fig-gradient-06.sage")

reset()

var('x,y,z')
f = x^2+y^2-z^2

# f= 0 : cone
G = implicit_plot3d(f-0.001,(x,-2,2),(y,-2,2),(z,-2,2),plot_points=150, color='darkkhaki')

#G.show(aspect_ratio=1, frame=False)

# f= 1 : hyperboloïde à une nappe
G = implicit_plot3d(f-0.3,(x,-2,2),(y,-2,2),(z,-2,2),plot_points=150, color='rosybrown')

G.show(aspect_ratio=1, frame=False)


# f= -1 : hyperboloïde à deux nappe
G = implicit_plot3d(f+0.4,(x,-2,2),(y,-2,2),(z,-2,2),plot_points=150, color='darkslategray')

#G.show(aspect_ratio=1, frame=False)


# Les trois sur la même figure 
G = implicit_plot3d(f-0.001,(x,-2,2),(y,-2,2),(z,-2,2),plot_points=150,color = 'darkkhaki',region=lambda x,y,z: x<=0.2 or y>=0.1 or (z<0.5 and z>-0.3))
G = G+implicit_plot3d(f-0.4,(x,-2,2),(y,-2,2),(z,-2,2),plot_points=150,color = 'rosybrown',region=lambda x,y,z: x<=0.1 or y>=0.2)
G = G+implicit_plot3d(f+0.4,(x,-2,2),(y,-2,2),(z,-2,2),plot_points=150,color = 'darkslategray')

#G.show(aspect_ratio=1, frame=False)


#G.show(aspect_ratio=1, frame=False, viewer='tachyon')
#G.save('fig-gradient-06.png')
