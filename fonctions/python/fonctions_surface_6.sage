
#load("fonctions_surface_6.sage")


var('x,y')

g = sqrt((x-1)**2+ (y-2)**2) + sqrt((x-3)**2 + (y-5)**2) + sqrt((x-6)**2 + (y-1)**2)

c = 6 # recherche sur [0,6]x[0,6]

T = 7  # Nb de doubles tranches
x0 = 0 # tranche initiale
for i in range(T):
	gy = g(x=x0)
	z,y0 = find_local_minimum(gy, 0, c)
	gx = g(y=y0)
	zz,x0 = find_local_minimum(gx, 0, c)
	print '---- Etape', i+1
	print 'y =',y0, ' z =',z	
	print 'x =',x0, ' z =',zz

# # Première tranche : x = 0
# print("--- Première tranche ---")
# x0 = 0
# g0 = g(x=x0)
# print(g0)
# z0,y0 = find_local_minimum(g0, 0, 6)
# print ' y0 =',y0, ' z =',z0
# # g0y = diff(g0,y)
# # sol = solve(g0y==0,y,solution_dict=True)
# # print 'y = ', sol[0]


# # Seconde tranche : y = y0
# print("--- Seconde tranche ---")
# gg0 = g(y=y0)
# print(gg0)
# zz0,x1 = find_local_minimum(gg0, 0, 6)
# print ' x1 =', x1, ' z =',zz0

