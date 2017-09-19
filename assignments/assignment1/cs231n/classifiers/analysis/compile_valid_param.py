
fac=open('Validation_accuracies2','r')
fp=open('Params_data2','r')


printed=False
cnt=0
while cnt<54:
	cnt+=1
	try:
		acc=fac.readline().strip()
		par=fp.readline().strip()
		#print "par",par
		
		acc=acc.split(':')[1].strip()
		params=par.split(',')[1:]
		mp={}
		mp['accuracy']=acc
		for param in params:
			key,value=param.split('=')
			mp[key]=value
		if not printed:
			printed=True
			for x in mp:
				print x,
			print
		for x in mp:
			print mp[x],
		print
	except:
		print "finished"
		break
fac.close()
fp.close()
	
	
