import requests
from scipy import stats
import h5py
import numpy as np
from velocity import get
import os
from scipy import interpolate
import pathlib
from simulation_data.galaxies import GalaxyPopulation
import scipy.linalg as la

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"47e1054245932c83855ab4b7af6a7df9"}


def particle_type(matter,id):
    
    
    
    
    redshift = 2
    scale_factor = 1.0 / (1+redshift)
    little_h = 0.6774
    url = "http://www.tng-project.org/api/TNG100-1/snapshots/z=" + str(redshift) + "/subhalos/" + str(id)


    if matter == "gas":
        part_type = 'PartType0'

    if matter == "stars":
        part_type = 'PartType4'

    params = {matter:'Coordinates,Masses'}

    sub = get(url)
    
    new_saved_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')
    
    if not os.path.exists(new_saved_filename):
        new_saved_filename = get(url+"/cutout.hdf5")
    
    
    effec_r = sub['halfmassrad_stars']*scale_factor
    
    
    
    with h5py.File(new_saved_filename,'r') as f:
        # NOTE! If the subhalo is near the edge of the box, you must take the 
        # periodic boundary into account! (we ignore it here)
        if(part_type == 'PartType0') and sub['mass_gas'] == 0:
            mass = np.zeros(999)
        elif(part_type == 'PartType4') and sub['mass_stars'] == 0:
            mass = np.zeros(999)
        else:
            dx = f[part_type]['Coordinates'][:,0] - sub['pos_x']
            dy = f[part_type]['Coordinates'][:,1] - sub['pos_y']
            dz = f[part_type]['Coordinates'][:,2] - sub['pos_z']
            masses = f[part_type]['Masses'][:]*(10**10 / 0.6774)
            rr = np.sqrt(dx**2 + dy**2 + dz**2)
            rr *= scale_factor/little_h # ckpc/h -> physical kpc
            
            r = np.where(rr < effec_r)
            
            mass,bin_edge,num = stats.binned_statistic(rr[r],masses,statistic='sum',bins=np.linspace(0,effec_r,1000))

        f.close()
        
        

    params = {'DM':'Coordinates,SubfindHsml'}
        
        
    with h5py.File(new_saved_filename,'r') as f:
        # NOTE! If the subhalo is near the edge of the box, you must take the 
        # periodic boundary into account! (we ignore it here)
        num = f['PartType1']['SubfindHsml'][:]

        dx = f['PartType1']['Coordinates'][:,0] - sub['pos_x']
        dy = f['PartType1']['Coordinates'][:,1] - sub['pos_y']
        dz = f['PartType1']['Coordinates'][:,2] - sub['pos_z']

        rr = np.sqrt(dx**2 + dy**2 + dz**2)
        rr *= scale_factor/little_h # ckpc/h -> physical kpc
        
        r = np.where(rr < effec_r)
        
        num_dm,bin_edge,x = stats.binned_statistic(rr[r],num,statistic='sum',bins=np.linspace(0,effec_r,1000))
        f.close()
        
        mass_dm_tot = 0.45*10**6
        mass_dm = num_dm*mass_dm_tot
        
        return mass_dm, mass,bin_edge


    
    
def find_circ_vel(id):
    
    redshift = 2
    scale_factor = 1.0 / (1+redshift)
    little_h = 0.6774
    url = "http://www.tng-project.org/api/TNG100-1/snapshots/z=" + str(redshift) + "/subhalos/" + str(id)
    
    g = 'gas'
    stars = 'stars'

    mass_dm,mass_gas,r_gas = particle_type(g,id)
    mass_dm,mass_stars,r_stars = particle_type(stars,id) 
    
    r = (r_gas[1:]+r_gas[:-1])/2

    G_const = 4.30091*10**-6


    mass_enc_gas = np.cumsum(mass_gas)
    mass_enc_stars = np.cumsum(mass_stars)
    mass_enc_dm = np.cumsum(mass_dm)

    mass_tot = mass_enc_dm + mass_enc_gas + mass_enc_stars

    vel_circ = np.sqrt(G_const*mass_tot/r)
    
    return r,vel_circ


def star_pos_vel(id):
    
    redshift = 2
    scale_factor = 1.0 / (1+redshift)
    little_h = 0.6774
    url = "http://www.tng-project.org/api/TNG100-1/snapshots/z=" + str(redshift) + "/subhalos/" + str(id)
    
    
    params = {'stars':'Coordinates,Velocities,GFM_StellarFormationTime'}

    sub = get(url)
    new_saved_filename = os.path.join('redshift_'+str(redshift)+'_data', 'cutout_'+str(id)+'_redshift_'+str(redshift)+'_rawdata.hdf5')
    if not os.path.exists(new_saved_filename):
        new_saved_filename = get(url+"/cutout.hdf5")
        
        
    with h5py.File(new_saved_filename,'r') as f:
        # NOTE! If the subhalo is near the edge of the box, you must take the 
        # periodic boundary into account! (we ignore it here)
        dx = (f['PartType4']['Coordinates'][:,0] - sub['pos_x'])*scale_factor
        dy = (f['PartType4']['Coordinates'][:,1] - sub['pos_y'])*scale_factor
        dz = (f['PartType4']['Coordinates'][:,2] - sub['pos_z'])*scale_factor
        
        dx_c = f['PartType4']['Coordinates'][:,0]
        dy_c = f['PartType4']['Coordinates'][:,1]
        
        vx = f['PartType4']['Velocities'][:,0]*np.sqrt(scale_factor) - sub['vel_x']
        vy = f['PartType4']['Velocities'][:,1]*np.sqrt(scale_factor) - sub['vel_y']
        vz = f['PartType4']['Velocities'][:,2]*np.sqrt(scale_factor) - sub['vel_z']

        star_masses = f['PartType4']['Masses'][:]*(10**10 / 0.6774)
        
        formation_time = np.array(f['PartType4']['GFM_StellarFormationTime'])
        
        select = np.where(formation_time > 0)[0]
        
        pos = np.array((dx,dy,dz)).T
        
        cor_pos = np.array((dx_c,dy_c)).T
        
        vel = np.array((vx,vy,vz)).T
        
    return(pos[select,:],vel[select,:],star_masses[select],cor_pos[select,:])

def rotational_data(id,string):
    
    redshift = 2
    scale_factor = 1.0 / (1+redshift)
    little_h = 0.6774
    url = "http://www.tng-project.org/api/TNG100-1/snapshots/z=" + str(redshift) + "/subhalos/" + str(id)
    sub = get(url)
    effec_r = sub['halfmassrad_stars']*scale_factor
    
    print(effec_r)
    r,vel_circ = find_circ_vel(id)
    
    pos,vel_raw,star_masses,cor_pos = star_pos_vel(id)
    
    radius = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    stars_select = np.where(radius < effec_r)[0]
    vel = np.array((vel_raw[stars_select, 0], vel_raw[stars_select, 1], vel_raw[stars_select, 2])).T
    rad = np.array((pos[stars_select, 0], pos[stars_select, 1], pos[stars_select, 2])).T
    mass = np.array((star_masses[stars_select],star_masses[stars_select],star_masses[stars_select])).T
    
    J_raw = mass*(np.cross(rad,vel))
    J = np.sum(J_raw,axis=0)
    J_mag = np.sqrt(np.dot(J,J))
    n_j = J/J_mag
    
    r_2d_sub = np.outer((np.dot(rad,n_j.T)),n_j)
    r_2d = rad - r_2d_sub
    r_2d_mag = np.sqrt(r_2d[:,0]*r_2d[:,0]+r_2d[:,1]*r_2d[:,1] + r_2d[:,2]*r_2d[:,2])
    n_r = np.array((r_2d[:,0]/r_2d_mag,r_2d[:,1]/r_2d_mag,r_2d[:,2]/r_2d_mag)).T
    
    n_phi = np.cross(n_j,n_r)
    
    v_phi = (vel[:,0]*n_phi[:,0] + vel[:,1]*n_phi[:,1] + vel[:,2]*n_phi[:,2])
    v_r = (vel[:,0]*n_r[:,0] + vel[:,1]*n_r[:,1] + vel[:,2]*n_r[:,2]) 
    v_j = np.dot(vel,n_j)
    
    v_final = ((vel[:,0]*vel[:,0] + vel[:,1]*vel[:,1] + vel[:,2]*vel[:,2]) - v_r**2 - v_j**2)
    
    velocity = np.sqrt((vel[:,0]*vel[:,0] + vel[:,1]*vel[:,1] + vel[:,2]*vel[:,2]))
    
    
    radius_new = np.sqrt((rad[:,0]*rad[:,0] + rad[:,1]*rad[:,1] + rad[:,2]*rad[:,2]))
    
    f = interpolate.interp1d(r, vel_circ,bounds_error = False,fill_value = 'extrapolate')
    new_v_circ = f(radius_new)
    
    e_v = v_phi/new_v_circ
    below = np.where(e_v < 0)[0]
    mass_below = sum(mass[below])
    mass_num = 2*mass_below/sum(mass)
    
    bins = np.linspace(-1.5,1.5,500)
    
    v_r_binned,r_test,x = stats.binned_statistic(radius_new,v_r,statistic='mean',bins=np.linspace(0,effec_r,50))
    r_binned = (r_test[1:]+r_test[:-1])/2
    
    lam_mean = np.mean(e_v) 
    
    #observed lambda
    
    n_z = np.array((0,0,1))
    
    lambda_obs = lam_mean * np.sqrt(1-np.dot(n_j, n_z)**2)
    
    cov  = np.cov(cor_pos.T)
    w, v = la.eig(cov)
    minor = np.min(np.float32(w))
    major = np.max(np.float32(w))
    ellipticity = 1.0-minor/major
    
    if str(string) == "lam":
        return lam_mean
    if str(string) == "bulge_ratio":
        return mass_num[0]
    if str(string) == "ellipticity":
        return ellipticity
    if str(string) == "lam_obs":
        return lambda_obs
    
    if str(string) == "":
        return r,vel_circ,v_r_binned,r_binned,e_v,bins,mass_num[0],v_j,radius_new,v_phi
    


def calc_lambda(ids):
    total_mass = np.zeros(len(ids))
    lambda_calc = np.zeros(len(ids))
    for i, id in enumerate(ids):
        lambda_calc[i] = rotational_data(id,'lam')
        print(str(i) + '/' + str(len(ids)))
    np.savetxt('z=2_Lambda', lambda_calc)
    lam = np.loadtxt('z=2_Lambda', dtype=float)
    return lam


def get_lambda(ids):
        file = pathlib.Path('z=2_Lambda')
        if file.exists ():
            lam = np.loadtxt('z=2_Lambda', dtype=float) 
            return lam
        else:
            return calc_lambda(ids)
        
def calc_bulge_ratio(ids):
    total_mass = np.zeros(len(ids))
    ratio_calc = np.zeros(len(ids))
    for i, id in enumerate(ids):
        ratio_calc[i] = rotational_data(id,'bulge_ratio')
        print(str(i) + '/' + str(len(ids)))
    np.savetxt('z=2_Ratio', ratio_calc)
    ratio = np.loadtxt('z=2_Ratio', dtype=float)
    return ratio


def get_bulge_ratio(ids):
        file = pathlib.Path('z=2_Ratio')
        if file.exists ():
            ratio = np.loadtxt('z=2_Ratio', dtype=float) 
            return ratio
        else:
            return calc_bulge_ratio(ids)

def calc_ellipticity(ids):
    total_mass = np.zeros(len(ids))
    ellipticity_calc = np.zeros(len(ids))
    for i, id in enumerate(ids):
        ellipticity_calc[i] = rotational_data(id,'ellipticity')
        print(str(i) + '/' + str(len(ids)))
        print("ID = " + str(id), "Ellipticity: " + str(ellipticity_calc[i]))
    np.savetxt('z=2_Ellipticity', ellipticity_calc)
    ellipticity = np.loadtxt('z=2_Ellipticity', dtype=float)
    return ellipticity
    
    
def get_ellipticity(ids):
    file = pathlib.Path('z=2_Ellipticity')
    if file.exists ():
        ratio = np.loadtxt('z=2_Ellipticity', dtype=float) 
        return ratio
    else:
        return calc_ellipticity(ids)
    
def calc_lam_obs(ids):
    total_mass = np.zeros(len(ids))
    lam_obs_calc = np.zeros(len(ids))
    for i, id in enumerate(ids):
        lam_obs_calc[i] = rotational_data(id,'lam_obs')
        print(str(i) + '/' + str(len(ids)))
        print("ID = " + str(id), "Lambda Observed: " + str(lam_obs_calc[i]))
    np.savetxt('z=2_Lambda_Obs', lam_obs_calc)
    lambda_obs = np.loadtxt('z=2_Lambda_Obs', dtype=float)
    return lambda_obs
    
    
def get_lam_obs(ids):
    file = pathlib.Path('z=2_Lambda_Obs')
    if file.exists ():
        lam_obs = np.loadtxt('z=2_Lambda_Obs', dtype=float) 
        return lab_obs
    else:
        return calc_lam_obs(ids)
        
        
#use to download the lambda and bulge_ratio values
#change to lambda functions and keys to download
def data_download():
    
    my_galaxy_population = GalaxyPopulation()
    
    with h5py.File('galaxy_population_data_'+str(2)+'.hdf5', 'r') as f:
        ids = f['ids'][:]
        median_age = f['median_age'][:]
        halfmass_radius = f['halfmass_radius'][:]
        total_mass = f['total_mass'][:]
        newbin_current_SFR = f['newbin_current_SFR'][:]
        maximum_merger_ratio_30kpc_current_fraction = f['maximum_merger_ratio_30kpc_current_fraction'][:]
        
    with h5py.File('galaxy_population_data_'+str(2)+'.hdf5', 'a') as f:
        #writing data
        d11 = f.create_dataset('bulge_ratio', data = get_bulge_ratio(ids))