
#load("fonctions_surface_4.sage")


var('x,y')

f = (x-1)**2+ (y-2)**2 + (x-3)**2 + (y-5)**2 + (x-6)**2 + (y-1)**2

print(f.expand())
# f = 3*x^2 + 3*y^2 - 20*x - 16*y + 76


fx = diff(f,x)
fy = diff(f,y)
sol = solve([fx==0,fy==0],(x,y),solution_dict=True)
print 'x,y = ', sol[0]
z = f(sol[0])
print ' z =', z
# x == 10/3, y == 8/3, z == 64/3





