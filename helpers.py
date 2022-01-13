import numpy as np
import pickle

nm  = 1852.                 # m    of 1 nautical mile

data = []

def load_trajectories():
    return pickle.load(open("trajectories.dat","rb"))

def conflict_detection(trajectories, thresholds,start_time=-1, end_time=-1):
    hor = thresholds[0]
    vert = thresholds[1]
    if start_time == -1 and end_time == -1:
        start_time = 0
        end_time = len(trajectories[0].time)
    
    conflicts = {}
    I = np.eye(len(trajectories))
    for idx in range(start_time, end_time):
        lats = np.array([t.lat[idx] for t in trajectories])
        lons = np.array([t.lon[idx] for t in trajectories])
        d = distance(np.asmatrix(lats), np.asmatrix(lons), np.asmatrix(lats), np.asmatrix(lons))
        d = np.asarray(d) + 1e9 * I
        
        alt = np.array([t.alt[idx] for t in trajectories])
        
        dalt = alt.reshape((1, len(trajectories))) - \
            alt.reshape((1, len(trajectories))).T  + 1e9 * I
        dalt = np.abs(dalt)
        
        conf_idx = np.where(np.logical_and(d <=hor, dalt <=vert))
        conf_idx = list(zip(conf_idx[0], conf_idx[1]))
        conf_idx = [tuple(x) for x in set(map(frozenset, conf_idx))]
        if len(conf_idx) > 0:
            conflicts[trajectories[0].time[idx]] = conf_idx
            
      
    return conflicts


def distance(lat1, lon1, lat2, lon2):
    """ Calculate distance vectors, using WGS'84
        In:
            latd1,lond1 en latd2, lond2 [deg] :positions 1 & 2 (vectors)
        Out:
            d [nm]    = distance from 1 to 2 in nm (matrix) """
    
    if not isinstance(lat1,np.ndarray):
        lat1 = np.array(lat1)
        lon1 = np.array(lon1)
        lat2 = np.array(lat2)
        lon2 = np.array(lon2)
    
    prodla =  lat1.T * lat2
    condition = prodla < 0

    r = np.zeros(prodla.shape)
    r = np.where(condition, r, rwgs84_matrix(lat1.T + lat2))

    a = 6378137.0

    r = np.where(np.invert(condition), r, (np.divide(np.multiply
      (0.5, ((np.multiply(abs(lat1), (rwgs84_matrix(lat1)+a))).T +
         np.multiply(abs(lat2), (rwgs84_matrix(lat2)+a)))),
            (abs(lat1)).T+(abs(lat2)+(lat1 == 0.)*0.000001))))  # different hemisphere

    diff_lat = lat2-lat1.T
    diff_lon = lon2-lon1.T

    sin1 = (np.radians(diff_lat))
    sin2 = (np.radians(diff_lon))

    sinlat1 = np.sin(np.radians(lat1))
    sinlat2 = np.sin(np.radians(lat2))
    coslat1 = np.cos(np.radians(lat1))
    coslat2 = np.cos(np.radians(lat2))

    sin21 = np.mat(np.sin(sin2))
    cos21 = np.mat(np.cos(sin2))
    y = np.multiply(sin21, coslat2)

    x1 = np.multiply(coslat1.T, sinlat2)

    x2 = np.multiply(sinlat1.T, coslat2)
    x3 = np.multiply(x2, cos21)
    x = x1-x3

    qdr = np.degrees(np.arctan2(y, x))

    sin10 = np.mat(np.abs(np.sin(sin1/2.)))
    sin20 = np.mat(np.abs(np.sin(sin2/2.)))
    sin1sin1 = np.multiply(sin10, sin10)
    sin2sin2 = np.multiply(sin20, sin20)
    sqrt = sin1sin1 + np.multiply((coslat1.T * coslat2), sin2sin2)
    dist_c = np.multiply(2., np.arctan2(np.sqrt(sqrt), np.sqrt(1-sqrt)))
    dist = np.multiply(r/nm, dist_c)
    #    dist = np.multiply(2.*r, np.arcsin(sqrt))
    
    return dist


def rwgs84_matrix(latd):
    """ Calculate the earths radius with WGS'84 geoid definition
        In:  lat [deg] (Vector of latitudes)
        Out: R   [m]   (Vector of radii) """

    lat    = np.radians(latd)
    a      = 6378137.0       # [m] Major semi-axis WGS-84
    b      = 6356752.314245  # [m] Minor semi-axis WGS-84
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    an     = a * a * coslat
    bn     = b * b * sinlat
    ad     = a * coslat
    bd     = b * sinlat

    anan   = np.multiply(an, an)
    bnbn   = np.multiply(bn, bn)
    adad   = np.multiply(ad, ad)
    bdbd   = np.multiply(bd, bd)
    # Calculate radius in meters
    r      = np.sqrt(np.divide(anan + bnbn, adad + bdbd))

    return r



class Traffic:
    '''
    Class definition
    Methods to implement:
        Traffic()    :    Constructor
        add()     :   Add aircraft to airspace
        delete()     :   Remove aircraft from airspace
        detect_conflict()    :    Give a list of conflicting aircraft
    
    Attributes:
        call_sign     :     ID of flight
        origin     :    Origin airport
        destination     :    destination airport
        ac_types    :    Type of aircraft
        flight_type    :    Manned or unmanned
        trajectories    :    Trajectories of present aircraft in the airspace
        safety_distances    :    Safety distances for this airspace
    '''
    
    def __init__(self):
        #pass
        self.call_sign = []
        self.origin = []
        self.destination = []
        self.trajectories = []
        self.safety_distances = (5,1000)
        
    def add(self,call_sign, origin, dest, trajectory):
        self.call_sign.append(call_sign)
        
        self.origin.append(origin)
        self.destination.append(dest)
        self.trajectories.append(trajectory)
        
        #pass
    def delete(self,idx):
        self.call_sign.pop(idx)
        self.origin.pop(idx)
        self.destination.pop(idx)
        self.trajectories.pop(idx)
        
        #pass
    def idx2id(self,idx):
        return self.call_sign.index(idx)
    
    def detect_conflicts(self):
        conflicts = conflict_detection(self.trajectories, self.safety_distances)
        #print(conflicts)
        
        conf_pairs = []
        for time, conf_list in conflicts.items():
            for pair in conf_list:
                if pair not in conf_pairs:
                    conf_pairs.append(pair)
        
        result = []
        for pair in conf_pairs:
            start = 0
            end = 0
            for time, conf_list in conflicts.items():
                if pair in conf_list:
                    if start == 0:
                        start = time
                        end = time
                    else:
                        end = time
            result.append((pair, start, end))
        
        return result
    
    
class Trajectory:
    '''
    Class definition
    Methods:
        Trajectory()    :    Constructor
        change_altitude()    :    Change the altitude of the trajectory
        flown_distance()    :    Determine the distance flown in this trajectory
    Attributes:
        time
        latitude
        longitude
        altitude
    '''
    
    def __init__(self,time,lat,long,alt):
        self.time = time
        self.lat = lat
        self.lon = long
        self.alt = alt
        
    def change_altitude(self,new_altitude, start_time, end_time):
        start = self.time.index(start_time)
        end = self.time.index(end_time)
        self.alt[start:end] = [new_altitude]*(end - start)
        #pass
    def flown_distance(self):
        dist = distance(self.lat[0], self.lon[0], self.lat[-1], self.lon[-1])
        return float(dist)