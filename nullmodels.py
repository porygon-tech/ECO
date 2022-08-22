
m,n = 100,120
strength = 0.7

a = np.zeros((m,n))
a[0,:] = np.random.choice((0,1),n)
for i in range(1,m):
	a=sort_degrees(a)
	a[i,:] = np.abs(a[i-1,:] - np.random.choice((0,1),n, p=(strength, 1-strength)))


showdata(a)
spectralRnorm(a)/a.sum()
spectralRnorm(np.triu(np.ones((m,n))))

#%%

m,n = 100,120
strength = 0.99

a = np.zeros((m,n))
f=0.6
a[0,:] = np.random.choice((0,1),n,p=(1-f,f))
a[:,0] = np.random.choice((0,1),m,p=(1-f,f))
for i in range(1,m):
    for j in range(1,n):
        #sort_degrees(a)
        p=(a[:i-1,j].sum()/(i) + a[i,:j-1].sum()/(j) )/2*strength+0.5*(1-strength)
        a[i,j] = np.random.choice((0,1), p=(1-p, p))

showdata(sort_degrees(a))
spectralR(a)



#%%
a = np.random.choice((0,1),(m,n), p=(0.7,0.3))
spectralRnorm(a)
a=np.triu(a)

showdata(a)
showdata(sort_degrees(a))
spectralRnorm(a)



spectralR(np.ones((m,n)))







#%%




showdata(a)
showdata()

