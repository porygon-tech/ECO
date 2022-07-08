



#------------------------------------
avg = []
c = pop.makeChildren(k=1)
for i in range(25):
	c = c.makeChildren(k=1)
	avg.append(c.avgPhenotype())



fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
x=np.arange(len(avg))
#ax.set_title(tickername + ', ' + str(np.round(npreview/60,2)) + ' hours preview')
ax.plot(x,avg, color='red')
#ax.plot(x,d[:npreview]+noisy[0], alpha=0.8)
#ax.plot(x,np.repeat(noisy[0], npreview), color='black')
plt.ylim(ps)
plt.show()


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def samplecolors(n, type='hex',palette=plt.cm.gnuplot):
	if type == 'hex':
		return list(map(plt.cm.colors.rgb2hex, list(map(palette, np.linspace(1,0,n)))))
	elif type == 'rgba':
		return list(map(palette, np.linspace(1,0,n)))



ntrials = 10
duration = 20
avg=np.zeros((ntrials,duration))
nindivs = 1000
nloci = 200
ps = (500,500+nloci)


skews=np.linspace(0.001,0.999,ntrials)
for i in range(ntrials):
	print('trial {0}'.format(i))
	skew = skews[i]
	pop = evo.population(nindivs,nloci, skew= skew,phenoSpace=ps);#pop.show()
	c = pop.makeChildren(k=1)
	for t in range(duration):
		print('\ttime {0}'.format(t))
		c = c.makeChildren(k=1)
		avg[i,t] = c.avgPhenotype()


fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
x=np.arange(duration)
ax.plot(x,avg.T)
plt.ylim(ps)
plt.show()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




nindivs = 1000
nloci = 200
ps = (500,500+nloci)
skew = 0.5
pop = evo.population(nindivs,nloci, skew= skew,phenoSpace=ps);#pop.show()
#pop.hist()

#-----------------------------------




















ntrials = 10
duration = 20
avg=np.zeros((ntrials,duration))
nindivs = 1000
nloci = 200
ps = (500,500+nloci)


skews=np.linspace(0.001,0.999,ntrials)
for i in range(ntrials):
	print('trial {0}'.format(i))
	skew = skews[i]
	pop = evo.population(nindivs,nloci, skew= skew,phenoSpace=ps);#pop.show()
	c = pop.makeChildren(k=1, mutRate=0.4)
	for t in range(duration):
		print('\ttime {0}'.format(t))
		c = c.makeChildren(k=1)
		avg[i,t] = c.avgPhenotype()


fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
x=np.arange(duration)
ax.plot(x,avg.T)
plt.ylim(ps)
plt.show()












pop.showfitness()
pop.showfitness(distbins=100)


nbins = nloci + 1
n, bins, patches = plt.hist(pop.phenotypes, 20, density=1, facecolor='g', alpha=0.7)
x=np.linspace(ps[0], ps[1], nloci)
y=evo.norm(bins,500+nloci*skew, np.sqrt(nloci*skew*(1-skew)))

plt.xlim(ps)
plt.plot(bins,y, '--', color='black')
plt.grid(True)
plt.show()


pop.set_fitnessLandscape('flat')
pop.set_sexualPreference('panmictic')
pop.set_intraCompetition('flat')
c = pop.makeChildren(k=1)



'''
nloci = 1000
skew = 0.7
s = np.random.binomial(nloci,skew,nindivs)

n, bins, patches = plt.hist(s, 100, density=1, facecolor='g', alpha=0.7)
x=np.linspace(ps[0], ps[1], nloci)
y=evo.norm(bins,nloci*skew, np.sqrt(nloci*skew*(1-skew)))

plt.plot(bins,y, '--')
plt.grid(True)
plt.show()
'''









	root = Path(".")
	my_path = root / 'data/obj'

	ntrials = 10
	duration = 100
	avg=np.zeros((ntrials,duration))
	nindivs = 1000
	nloci = 200
	ps = (500,500+nloci)


	def f(x):
		return (x-ps[0]) / (ps[1]-ps[0]) # linear 

	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	#pop = evo.population(nindivs,nloci, skew= 0.5,phenoSpace=ps);#pop.show()
	#pop.set_fitnessLandscape(f)
	#pop.showfitness(20)
	if ntrials > 1:
		skews=np.linspace(0.001,0.999,ntrials)
	else:
		skews=[0.25]

	for i in range(ntrials):
		print('trial {0}'.format(i))
		skew = skews[i]
		pop = evo.population(nindivs,nloci, skew= skew,phenoSpace=ps);#pop.show()
		pop.set_fitnessLandscape(f)
		c = pop.makeChildren(k=1, mutRate=0)
		for t in range(duration):
			print('\ttime {0}'.format(t))
			c = c.makeChildren(k=1)
			avg[i,t] = c.avgPhenotype()

	'''
	fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
	x=np.arange(duration)
	ax.plot(x,avg.T)
	plt.ylim(ps)
	plt.show()


	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


	d = np.diff(avg, axis=1)

	fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
	ax.scatter(avg[:,:-1].flatten(), d.flatten())
	plt.show()
	'''










	







def show3D(func, rangeX=[0,10],rangeY=[0,10],color=plt.cm.jet, resolution=50):
	resX=resY=resolution
	x = np.linspace(rangeX[0],rangeX[1],resX)
	y = np.linspace(rangeY[0],rangeY[1],resY)
	gx,gy = np.meshgrid(x,y)
	x, y = gx.flatten(), gy.flatten()
	z = func(x,y)
	fig = plt.figure(); ax = fig.add_subplot(projection='3d')
	surf = ax.plot_trisurf(x,y,z, cmap=color, linewidth=0)
	fig.colorbar(surf)
	plt.show()




def sexualPreference(x,y,k=1):
	return 1-1/(1+(x-y)**2/k) # the more different, the more attractive


show3D(sexualPreference)

def f(x,y):
	return sexualPreference(x,y, k=5)

show3D(f)
