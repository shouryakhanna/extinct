exec(open("./packages.py").read()) 
import numpy as np
import numpy
import os, sys, time,  math
from time import sleep
import healpy as hp
import ebf
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib  as mpl	

# dustmaploc = '/iranet/users/pleia14/Documents/pdoc_work/dustmaps'




def extmapsanj(l,b,r,GalaxiaPath):
	
	'''
	function to interpolate extinction from 2D and 3D maps
			
    NAME: extmapsanj

    PURPOSE: obtain ebv using Sanjib's interpolated extinction map

    INPUT: l,b,r

    OUTPUT: ebv-2d, ebv-3d, intfac
       
    HISTORY: August 16, 2022
	
	'''	
	l = np.array(l)
	b = np.array(b)
	r = np.array(r)

	

	data = {}
	data['glon'] = l.copy()
	data['glat'] = b.copy()
	data['rad'] = r.copy()
	
		
	import scipy.interpolate

	GalaxiaPath = GalaxiaPath+'/'
	# GalaxiaPath=getdirec('galaxia')+'/Extinction/'
	x=np.zeros((data['glon'].size,3),dtype='float64')
	x[:,0]=data['glon']
	x[:,1]=data['glat']
	x[:,2]=np.log10(data['rad'])
	data3d=ebf.read(GalaxiaPath+'ExMap3d_1024.ebf','/ExMap3d.data')
	xmms3d=ebf.read(GalaxiaPath+'ExMap3d_1024.ebf','/ExMap3d.xmms')
	points3d=[xmms3d[i,0]+np.arange(data3d.shape[i])*xmms3d[i,2] for i in range(data3d.ndim)]
	data2d=ebf.read(GalaxiaPath+'Schlegel_4096.ebf','/ExMap2d.data')
	xmms2d=ebf.read(GalaxiaPath+'Schlegel_4096.ebf','/ExMap2d.xmms')
	points2d=[xmms2d[i,0]+np.arange(data2d.shape[i])*xmms2d[i,2] for i in range(data2d.ndim)]
	
	temp3d=scipy.interpolate.interpn(points3d,data3d,x,bounds_error=False,fill_value=None,method='linear')
	temp2d=scipy.interpolate.interpn(points2d,data2d,x[:,0:2],bounds_error=False,fill_value=None,method='linear')
	data['exbv_schlegel_inf']=temp2d
	data['exbv_schlegel']=temp2d*temp3d	
	
	return 	temp2d, temp2d*temp3d, temp3d	

class dustmap_green(object):

	'''
	estimate extinction from 3d maps from Green et al. 2019 using their downloaded map
	l (degrees)
	b (degrees)
	d (kpc)
	
	machine: home or huygens
	'''	
	def __init__(self,dustmaploc='',usemap='bayestar'):
		print('')
		
		self.usemap = usemap				
		if dustmaploc == '':
			raise RuntimeError('no dustmaploc provided')			
		self.readloc_ = dustmaploc		

					
	def loadit(self,max_samples=20):
						
		from dustmaps.bayestar import BayestarQuery
		# timeit=stopwatch()
		# timeit.begin()
			
		maploc = self.usemap
		mapname = os.listdir(self.readloc_+'/'+maploc)[0]
		
		self.bayestar = BayestarQuery(max_samples=max_samples,map_fname=self.readloc_+'/'+maploc+'/'+mapname)


		# timeit.end()	
		return 
		
	def get(self,l,b,d,band=None,mode='best',getebv_only=False,getebv=False,pcent=False):
		
		#mode='median'
		
		from astropy.coordinates import SkyCoord
		import astropy.units as u		
	
		l = l*u.deg
		b = b*u.deg
		d = (d*1000.)*u.pc
		
		coords = SkyCoord(l, b, distance=d, frame='galactic')
		
		q = self.bayestar 
		
		E = q(coords, mode=mode)
		if pcent:
			E_ = q(coords, mode='percentile',pct=[16,50,84])		
			# E_unc = np.array([np.mean([E_[i][1] - E_[i][0],E_[i][2] - E_[i][1]]) for i in range(len(E_))])
			E_unc = np.array([0.5*(E_[i][2] - E_[i][0]) for i in range(len(E_))])   
		
		# if mode == 'median':
			# E = q(coords, mode=mode)
		# elif mode == 'samples':
			# E = q(coords, mode='samples')	
			
		if getebv_only:
			
			return E, E_unc
		
		else:
			
			rfac = redden_fac(maptype='bayestar3d')
			
			if band in rfac.keys():				
				A_lambda = E*rfac[band]
			elif l.size == 1:
				A_lambda = np.nan
			else:
				A_lambda = np.zeros(len(l)) + np.nan
	
			if getebv:
				print(' A_lambda & E(B-V) ')
				return A_lambda, E		
			else:
				return A_lambda
class dustmap_decaps(object):

	'''
	estimate extinction from 3d maps from Zucker et al. 2025 using their downloaded map
	l (degrees)
	b (degrees)
	d (kpc)
	https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/J9JCKO
	machine: home or huygens
	'''	
	def __init__(self,dustmaploc=''):
		print('')
		
		from dustmaps.decaps import DECaPSQueryLite 
		
		dcp = DECaPSQueryLite(map_fname=dustmaploc+'/decaps/decaps/decaps_mean.h5')
		self.dquery = dcp.query
	
	def get(self,l,b,d,band=None,getebv_only=False,getebv=False,pcent=False):
		
		#mode='median'
	
		from astropy.coordinates import SkyCoord
		import astropy.units as u		
	
		l = l*u.deg
		b = b*u.deg
		d = (d*1000.)*u.pc
		
		coords = SkyCoord(l, b, distance=d, frame='galactic')
		self.coords = coords
		
		E = self.dquery(coords)

		return E
		
class lallement_maps(object):

	'''
	estimate extinction from 3d maps from various lallement_maps
	l (degrees)
	b (degrees)
	d (kpc)
	

	version = 19, 22

	Usage:	
		

	llm = lallement_maps(dustmaploc=pdocdir+'/dustmaps',useprecomp=False); 		
	llm.setup(maplists[inum],resol=dresols[inum])

	precomputed maps (at healpix level 7)
	
	
	'''	
	def __init__(self,dustmaploc='',useprecomp=False):
		print('')
				
		if dustmaploc == '':
			raise RuntimeError('no dustmaploc provided')			
		self.dustmaploc = dustmaploc			
		self.getlist()		
		self.useprecomp = useprecomp
		self.hplevel = 7 # pre computed map
		self.Rv = 3.1 #default option
	
	def getlist(self):
		'''
		lists out the available maps

		# lallement 2019
		# https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/625/A135
		
		#lallement 2022
		# https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/661/A147#/browse
		
		# vergely 2022
		# https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/664/A174	

		
		'''
				
		self.mapnames = ['lallement2019','lallement2022','vergely2022']
		self.resolutions = ['lallement2019 ()','lallement2022 ()','vergely2022 (10pc, 25pc, 50pc)']

		print('')
		print('------------------------------------------------')
		# print('List of available models:')
		dfmodlist = pd.DataFrame(self.resolutions,columns=['Lallement extinction models / resolutions:'])
		print(dfmodlist)
		print('------------------------------------------------')
	

	def setup(self,mapuse,resol=''):
		
		'''
		
		fix dmax for precomputed version!!!!
		have set it to 15 kpc for now
		'''
		
		self.resol = resol
		self.mapuse = mapuse

		from astropy.io import fits
		if '19' in mapuse:
			self.fname = 'map3D_GAIAdr2_feb2019.h5'			
			self.fl = h5py.File(self.dustmaploc+'/'+mapuse+'/'+self.fname, 'r')
			self.dcube = self.fl['stilism']['cube_datas'][:]
			self.header_present = False
		if mapuse == 'lallement2022':
			self.fname = 'cube_ext.fits'
			self.fl = fits.open(self.dustmaploc+'/'+mapuse+'/'+self.fname); self.dcube = self.fl[0].data.T 		
			self.header_present = True	
		if 'verg' in mapuse:
			self.fname = 'explore_cube_density_values_0'+str(self.resol)+'_v2.fits'
			self.fl = fits.open(self.dustmaploc+'/'+mapuse+'/'+self.fname); self.dcube = self.fl[0].data.T 
			self.header_present = True			



		fl = self.fl
		dcube = self.dcube


		if self.header_present:
			abs_xmax = int((fl[0].header['NAXIS1'] - 1)*fl[0].header['STEP']/2.)
			abs_ymax = int((fl[0].header['NAXIS2'] - 1)*fl[0].header['STEP']/2.)
			abs_zmax = int((fl[0].header['NAXIS3'] - 1)*fl[0].header['STEP']/2.)
			grid_step = fl[0].header['STEP']
			xval_ = np.arange(0.,fl[0].header['NAXIS1'])
			yval_ = np.arange(0.,fl[0].header['NAXIS2'])
			zval_ = np.arange(0.,fl[0].header['NAXIS3'])
			self.zpix_cen = math.floor(fl[0].header['SUN_POSZ'])
			self.xpix_cen = math.floor(fl[0].header['SUN_POSX'])
			self.ypix_cen = math.floor(fl[0].header['SUN_POSY'])
		else:
			# for the lallement2019 case
			abs_xmax = int((1201 - 1)*5./2.)
			abs_ymax = int((1201 - 1)*5./2.)
			abs_zmax = int((161 - 1)*5./2.)
			grid_step = 5.
			xval_ = np.arange(0.,1201)
			yval_ = np.arange(0.,1201)
			zval_ = np.arange(0.,161)
			self.zpix_cen = math.floor(80.5)
			self.xpix_cen = math.floor(600.5)
			self.ypix_cen = math.floor(600.5)			
			

		xmin,xmax = -abs_xmax,abs_xmax
		ymin,ymax = -abs_ymax,abs_ymax
		zmin,zmax = -abs_zmax,abs_zmax
	


		xvals_ = xmin + grid_step*xval_
		yvals_ = ymin + grid_step*yval_
		zvals_ = zmin + grid_step*zval_
		
	
		xv1,yv1 = np.meshgrid(xval_,yval_,indexing='ij')
		xv = xv1.flatten(); yv = yv1.flatten()
		xv = xmin + grid_step*xv
		yv = ymin + grid_step*yv


		xpnt = xmin + xval_*grid_step
		zpnt = zmin + zval_*grid_step
		
				
		self.xSun = 0.
		self.ySun = 0.
		self.zSun = 0.		
		
		self.xvals_ = xvals_
		self.yvals_ = yvals_
		self.zvals_ = zvals_
		

		
		
		self.grid_step = grid_step			
		self.dmax = (dtools.sqrtsum(ds=[xmax,ymax,zmax]))  #[pc]
		self.dgrid = np.linspace(0.01,self.dmax,int((self.dmax/self.grid_step)*5.))  # [pc]
		self.xmin = xmin;self.xmax = xmax
		self.ymin = ymin;self.ymax = ymax
		self.zmin = zmin;self.zmax = zmax

		if self.useprecomp:			
			self.fl_pre = dtools.pickleread(self.dustmaploc+'/'+self.mapuse+'/intp_'+self.mapuse+'_'+self.resol+'.pkl')		
			self.dres = (float(self.resol.split('pc')[0])/5.)/1000.
		
			self.dmax_guess = 105.	
			self.dgrid_pre = np.linspace(0.,self.dmax_guess,int(self.dmax_guess/self.dres))
			
			self.a0map = np.array([self.fl_pre['func'][i](self.dgrid_pre) for i in range(self.fl_pre['func'].size)])		

			self.hpinfo = dtools.hpinfo(self.hplevel)
			

	def getvals(self,l,b,duse=1,getlos=False):
		
		'''
		uses the full data cube
		
		l    [deg]
		b    [deg]
		duse [kpc]

		
		if getlos:			
			return d_kpc, A0(d), E(B-V)(d) = A0(d)/self.Rv
		if getlos == False:
			return A0(d_kpc), E(B-V)/self.Rv
			
		'''
		
		if self.useprecomp == False:				

			dmock = {}	
			dmock['x1'],dmock['y1'],dmock['z1'] = autil.lbr2xyz(l,b,self.dgrid)
					
			
			points = (self.xvals_,self.yvals_,self.zvals_)
			values = self.dcube.copy()	
			
			fac = 1./((self.dgrid[1]-self.dgrid[0]))
			vals = []
			for i in range(dmock['x1'].size):
				point = np.array([dmock['x1'][i],dmock['y1'][i], dmock['z1'][i]])
				vals.append(interpn(points, values, point,bounds_error=False,fill_value=None,method='linear')[0])
			
			vals = np.array(vals)						
			A0_ = np.cumsum(vals)/fac
			
			idgrid = np.arange(0,self.dgrid.size)
			# f = interp1d(self.dgrid,idgrid,bounds_error=False,fill_value=(idgrid[0],idgrid[1])) # interpolation done on parsec grid
			f = interp1d(self.dgrid/1000.,idgrid,bounds_error=False,fill_value=(idgrid[0],idgrid[1]))   # interpolation done on kiloparsec grid
	
			indbox = np.where((abs(dmock['x1'] - self.xSun )<self.xmax/2.)&(abs(dmock['y1']- self.ySun/2. )<self.ymax)&(abs(dmock['z1'] - self.zSun )<self.zmax))[0]
		
			if getlos:
				return self.dgrid[indbox]/1000.,A0_[indbox], A0_[indbox]/self.Rv
			if getlos == False and ((duse*1000.) < np.nanmax(self.dgrid[indbox])) :				
				return A0_[int(f(duse))], A0_[int(f(duse))]/self.Rv  #*self.grid_step
			if getlos == False and ((duse*1000.) > np.nanmax(self.dgrid[indbox])) :				
				return np.nan

		if self.useprecomp:			

			dindx = (duse/self.dres).astype(int)
			ipix = hp.ang2pix(self.hpinfo['nside'],l,b,lonlat=True)
			

			indl,indr,indnm = tabpy.crossmatch(self.fl_pre['hpix_id'],ipix)
			a0val_tmp = self.a0map[indl,dindx]			
		

			self.ipix = ipix
			self.dindx = dindx
		
			a0val = np.zeros(duse.size) + np.nan 
			dmock = {}	
			dmock['x1'],dmock['y1'],dmock['z1'] = autil.lbr2xyz(l,b,duse)			 
			indbox = np.where((abs(dmock['x1']*1000. - self.xSun )<self.xmax/2.)&(abs(dmock['y1']*1000. - self.ySun/2. )<self.ymax)&(abs(dmock['z1']*1000. - self.zSun )<self.zmax))[0]					
			

			a0val[indbox] = a0val_tmp[indbox].copy()

			return a0val, a0val/self.Rv
			


class extmap():
	
	
	def __init__(self,dustmaploc='',figloc='---',renorm=True,print_=False,load_bstar=False,load_marshall=False):
		
		'''
		notes for user:
			mwdust requires entries as floats
		
		to add: 
				my intfac option for calculation
				de-redden factors
				interpolation at boundaries
				
				
				
				
		'''	
		self.print_ = print_
	
		self.mkfit = False
		self.priority = {0:'lallement',1:'bayestar',2:'schlegel_3d',3:'schlegel_2d'}
		self.dustmaploc = dustmaploc
		self.GalaxiaPath = dustmaploc+'/galaxia'
		

		# self.extloc = '/iranet/users/pleia14/Documents/pdoc_work/science/extinction_work'
		self.extloc = dustmaploc.split('/dustmaps')[0]+'/science/extinction_work'
		self.jiefname = 'sedpair_extinction_use.fits'



		
		self.load_bstar=load_bstar
		self.load_marshall=load_marshall

		self.loadmaps()

		self.renorm = renorm
		if renorm:		
			self.correc_fac_bayestar = 0.884  # correction factor for Bayestar 2019 E(B-V)
			self.correc_fac_schlegel = 1.0 # correction factor for recalibrated normalisation (14% lower) of E(B-V) from SFD 1999 [see Schlafly 2010]
			self.sfdrescale = True
		else:
			self.correc_fac_bayestar = 1
			self.correc_fac_schlegel = 1
			self.sfdrescale = False


	def loadmaps(self):
		
		import mwdust
		print('imported mwdust..')

		lmap = lallement_maps(dustmaploc=self.dustmaploc,useprecomp=True); 		
		lmap.setup('vergely2022',resol='25pc')
		self.lmap = lmap

		if self.load_bstar:
			print('load_bstar = '+str(self.load_bstar))
			bstar=dustmap_green(dustmaploc=self.dustmaploc)
			bstar.loadit() 	
			self.bstar = bstar
		if self.load_marshall:
			print('load_marshall = '+str(self.load_marshall))
			self.marshall = mwdust.Marshall06(filter='E(B-V)')			
			
		

		self.combined19= mwdust.Combined19(filter='E(B-V)')
		self.drimmel = mwdust.Drimmel03(filter='E(B-V)')

		self.green19 = mwdust.Green19(filter='E(B-V)')
		
		# self.fjie = tabpy.read(self.dustmaploc+'/jie_extinct2023.fits')
		self.fjie = tabpy.read(self.extloc+'/'+self.jiefname)
	def sfd_rescale(self,ebv):
		'''
		rescale 2d sfd where ebv > 0.15	
		'''	
		if self.sfdrescale:
			indcor = np.where(ebv > 0.15)[0]	
			if self.print_:
				print('fixing sfd for ...')
				print(indcor.size)
				print('')
			cor_fac = 0.6 + (0.2*(1 - np.tanh((ebv[indcor] - 0.15)/0.3) ) )		
			ebv[indcor] = ebv[indcor]*cor_fac		
		
		return ebv

	def func_marshall(self,l,b,d):
		'''
		problem!
		'''
		val = np.nan
		try:
			val = self.marshall(l[0],b[0],d)
			return val
		except IndexError:
			pass
			
		return val		

	def pickleread(self,file1):
		
		'''
		read pickle files
		input: fileloc+'/'+filename
		
		'''
		import pickle	
		data = pickle.load(open(file1,'rb'))	
		
		
		return data
	


	def picklewrite(self,data,nam,loc,prnt=True):
		'''
		write files using pickle
		'''
		print('')
		
		
		import pickle
		
		pickle.dump(data,open(loc+'/'+nam+'.pkl','wb'))	
		if prnt:
			print(nam+' .pkl written to '+loc)	
			
		return 


	def getebv_options(self):
		
		typs = ['3d','3d_test','lallement','sfd','sfd_rescaled_binney','my','bstar']
		
		print(typs)
		
	def getebv(self,l,b,d,typ='3d',prnt_=True):

		'''
		typs = ['3d','3d_test','lallement','sfd','sfd_rescaled_binney','my','bstar']
		
		'''
		
		if type(l) != numpy.ndarray:
			raise RuntimeError('please provide l,b,d as np.array')
	
		if typ == '3d':
			
			'''
			lallement - > green -> dust_model -> sfd_corr
			'''

			iproc = 0
			
			A0,ebv = self.llm.getvals(l,b,duse=d)
			indf = np.where(np.isfinite(ebv))[0]
			indnan = np.where(np.isnan(ebv))[0]
			if indnan.size > 0:
				
				if prnt_:
					print('using '+self.priority[iproc]+' for '+str(indf.size)+' stars')
				

				ebv_corr =self.green19(l[indnan],b[indnan],d[indnan])
				ebv[indnan] = ebv_corr				
				indf = np.where(np.isfinite(ebv_corr))[0]
				
				if indf.size > 0:	
					
					if prnt_:
						print('using '+self.priority[iproc+1]+' for '+str(indf.size)+' stars')
				
				indnan = np.where(np.isnan(ebv))[0]
				if indnan.size > 0:		
					if 	prnt_:
						print('using '+self.priority[iproc+2]+' for '+str(indnan.size)+' stars')

					ebv_corr = extmapsanj(l[indnan],b[indnan],d[indnan],self.GalaxiaPath)[0]										
					ebv_corr = self.sfd_rescale(ebv_corr)*self.correc_fac_schlegel					
					intfac_ = extmapsanj(l[indnan],b[indnan],d[indnan],self.GalaxiaPath)[2]					
					ebv[indnan] = ebv_corr*intfac_
					
			return ebv

				
		if typ == 'lallement':					
			ebv = self.lmap.getvals(l,b,d)			
			return ebv
			
		if typ == 'sfd':					
			ebv = extmapsanj(l,b,d,self.GalaxiaPath)[0]				
			# print('using '+self.priority[3]+' for '+str(l.size)+' stars')
			return ebv

		if typ == 'sfd_rescaled_binney':		
			
			ebv = extmapsanj(l,b,d,self.GalaxiaPath)[0]
			ebv = self.sfd_rescale(ebv)*self.correc_fac_schlegel	
			# print('using '+self.priority[3]+' for '+str(l.size)+' stars')
			return ebv
			
		if typ == 'my':		

			ebv = extmapsanj(l,b,d,self.GalaxiaPath)[0]
			ebv = self.sfd_rescale(ebv)*self.correc_fac_schlegel	
			intfac_ = extmapsanj(l,b,d,self.GalaxiaPath)[2]					
			ebv = ebv*intfac_
			return ebv
			
		if typ == 'bstar':
					
			ebv_pack = self.bstar.get(l,b,d,getebv_only=True,pcent=True)		
			ebv,ebv_unc =ebv_pack[0]*self.correc_fac_bayestar, ebv_pack[1]
			
			return ebv,ebv_unc		


