# Tangente à une cubique
reset()

var('x,y')
f = x^3-y^2-x
G = implicit_plot(f,(x,-2,2),(y,-2,2),contours=srange(-0.8,1,0.2),color='black',linewidth=1.2)
G = G+implicit_plot(f,(x,-2,2),(y,-2,2),contours=srange(-3.3,-1,0.5),color='black',linewidth=1.2)
G = G+implicit_plot(f,(x,-2,2),(y,-2,2),contours=srange(1,2,0.5),color='black',linewidth=1.2)

x0 = sqrt(3)/3; y0 =0
k0 = f(x=x0,y=y0)
G = G+implicit_plot(f,(x,-2,2),(y,-2,2),contours=[k0],color='green',linewidth=2)
G = G+implicit_plot(f,(x,-2,2),(y,-2,2),contours=[-k0-0.001],color='red',linewidth=2)
G.show()

G.save('fig-gradient-04.png')
