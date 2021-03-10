import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.units import degrees, nautical
from collections import defaultdict
from sklearn.cluster import DBSCAN
import datetime
from astropy.timeseries import LombScargle

def number_location_transitions(g, loc_label_col="location_label"):
    if g is None or len(g) == 0:
        return None
    # ignores transitions from moving to static and vice-versa, but counts transitions from outliers to major location clusters
    count = 0
    prev_loc_label = None
    for row in g.iterrows():
        cur_loc_label = row[1][loc_label_col]
        if np.isnan(cur_loc_label):
            continue
        elif prev_loc_label == None :
            prev_loc_label = int(cur_loc_label)
        else:
            if prev_loc_label != int(cur_loc_label):
                count += 1
                prev_loc_label = int(cur_loc_label)
    return count

def len_stay_at_clusters_in_minutes(g, SAMPLE_RATE, loc_label_col="location_label"):
    if g is None or len(g) == 0:
        return pd.Series({"max_len_stay_at_clusters_in_minutes": None, "min_len_stay_at_clusters_in_minutes": None, "std_len_stay_at_clusters_in_minutes": None, "mean_len_stay_at_clusters_in_minutes": None})
    lenstays = []
    count = 0
    prev_loc_label = None
    for row in g.iterrows():
        cur_loc_label = row[1][loc_label_col]
        if np.isnan(cur_loc_label):
            continue
        elif prev_loc_label == None :
            prev_loc_label = int(cur_loc_label)
            count += 1
        else:
            if prev_loc_label == int(cur_loc_label):
                count += 1
            else:
                lenstays.append(count)
                prev_loc_label = int(cur_loc_label)
                count = 0+1
    if count>0: # in case of no transition
        lenstays.append(count)
    lenstays  = np.array(lenstays)*SAMPLE_RATE
    #print len(lenstays)
    if len(lenstays)>0:
        smax = np.max(lenstays)
        smin = np.min(lenstays)
        sstd = np.std(lenstays)
        smean = np.mean(lenstays)
    else:
        smax = None
        smin = None
        sstd = None
        smean = None
    return pd.Series({"max_len_stay_at_clusters_in_minutes": smax, "min_len_stay_at_clusters_in_minutes": smin, "std_len_stay_at_clusters_in_minutes": sstd, "mean_len_stay_at_clusters_in_minutes": smean})

def distance_row(x):
    """
    :param x: A row from a dataframe
    :return: The distance in meters between
    """
    if np.isnan(x['_lat_before']) or np.isnan(x['_lon_before']) or np.isnan(x['_lat_after']) or np.isnan(x['_lon_after']):
        return 0
    try:
        return geodesic((x['_lat_before'], x['_lon_before']),
                 (x['_lat_after'], x['_lon_after'])).meters
    except UnboundLocalError:
        return 0

def get_all_travel_distances_meters(g, SAMPLE_RATE, lat_col="latitude", lng_col="longitude", loc_label_col="location_label"):
    if g is None or len(g) == 0:
        return None
    lat_lon_temp = pd.DataFrame()

    lat_lon_temp['_lat_before'] = g[lat_col]
    lat_lon_temp['_lat_after'] =  g[lat_col].shift(-1)
    lat_lon_temp['_lon_before'] = g[lng_col]
    lat_lon_temp['_lon_after'] =  g[lng_col].shift(-1)
    lat_lon_temp["location_label"] = g[loc_label_col]
    lat_lon_temp['time_before'] = g.index
    lat_lon_temp['time_after'] = lat_lon_temp['time_before'].shift(-1)
    lat_lon_temp['time_diff'] = lat_lon_temp['time_after'] - lat_lon_temp['time_before']

    time_okay = (lat_lon_temp['time_diff']==pd.Timedelta(str(SAMPLE_RATE)+"min"))
    changes_selector = (time_okay)
    distances = lat_lon_temp.apply(distance_row, axis = 1)[changes_selector]
    return distances

def travel_distance_meters(g, SAMPLE_RATE): # to be computed on static and moving both
    ## Distance will not be computed over gaps larger than "SAMPLE_RATE" number of minutes
    ## This is done to enable computation of traveled distance for day1_night+day2_night+....
    ## A jump between person's location from day1_night to day2_night is not unusual. Hence, we want to ignore that.
    ## Remember to change SAMPLE_RATE on the top
    if g is None or len(g) == 0:
        return None
    distances = get_all_travel_distances_meters(g, SAMPLE_RATE)
    total_distance = distances.sum()
    return total_distance

def travel_distance_and_related_meters(g):
    if g is None or len(g) == 0:
        return pd.Series({"total_distance_meters": None, "speed_mean_meters_per_sec": None,"speed_var_meters_per_sec": None})
    distances = get_all_travel_distances_meters(g)
    total_distance = distances.sum()
    spd_in_meters_per_sec = distances.div(SAMPLE_RATE*60)
    spd_mean = spd_in_meters_per_sec.mean()
    spd_var = spd_in_meters_per_sec.var()
    rdict = {"total_distance_meters": [total_distance], "speed_mean_meters_per_sec": [spd_mean], "speed_var_meters_per_sec": [spd_var]}
    #return rdict
    return pd.Series({"total_distance_meters": total_distance, "speed_mean_meters_per_sec": spd_mean, "speed_var_meters_per_sec": spd_var})

def radius_of_gyration(g):
    if g is None or len(g) == 0:
        return None
    #Center is the centroid, nor the home location
    not_noise = g[g["location_label"] != -1]
    changes_selector = (not_noise["location_label"].shift() != not_noise["location_label"])
    mobility_trace = not_noise[changes_selector]

    #Average x,y
    lat_lon = mobility_trace[["latitude","longitude"]].values
    center = np.average(lat_lon, axis = 0)
    norm = np.linalg.norm(lat_lon - center)
    return np.sqrt(norm) / len(lat_lon)


def circadian_movement_energies(g, timestamp_ms_col, lat_col, lng_col):
    t = (g[timestamp_ms_col].values/1000.0) # seconds
    ylat = g[lat_col].values
    ylong = g[lng_col].values
    pHrs = np.arange(23.5,24.51, 0.01) # hours
    pSecs = pHrs*60*60 # seconds
    f = 1/pSecs

    pgram_lat = LombScargle(t, ylat).power(frequency = f, normalization='psd')
    pgram_long = LombScargle(t, ylong).power(frequency = f, normalization='psd')

    E_lat = np.sum(pgram_lat)
    E_long = np.sum(pgram_long)
    return (E_lat, E_long)

def circadian_movement(g, timestamp_ms_col="timestamp", lat_col="latitude", lng_col="longitude"):
    if g is None or len(g) == 0:
        return None
    E_lat, E_long = circadian_movement_energies(g, timestamp_ms_col, lat_col, lng_col)
    return np.log10(E_lat+E_long)

def location_entropy(g, loc_label_col="location_label"):
    if g is None or len(g) == 0:
        return None
    g =  g.dropna(how='any') # should not be required if input is static
    g = g.drop(g[(g[loc_label_col] < 1)].index) # remove outliers/ cluster noise
    if len(g)>0:
        #Get percentages for each location
        percents = g[loc_label_col].value_counts(normalize= True)
        entropy = -1 * percents.map(lambda x: x * np.log(x)).sum()
        return entropy
    else:
        return None

def location_entropy_normalized(g, loc_label_col="location_label"):
    if g is None or len(g) == 0:
        return None
    g =  g.dropna(how='any') # should not be required if input is static
    g = g.drop(g[(g[loc_label_col] < 1)].index)  # remove outliers/ cluster noise
    entropy = location_entropy(g)
    unique_clusters = g[loc_label_col].unique()
    num_clusters = len(unique_clusters)
    if num_clusters ==0 or len(g) == 0 or entropy is None:
        return None
    else:
        return entropy/num_clusters
    
def location_variance(g, lat_col="latitude", lng_col="longitude"):
    if g is None or len(g) == 0:
        return None
    g =  g.dropna(how='any') # should not be required if input is static
    lat_var = g[lat_col].var()
    long_var = g[lng_col].var()
    if len(g)>0:
        return (lat_var+long_var)
    else:
        return None

def location_variance_log(g, lat_col="latitude", lng_col="longitude"):
    tiny_val = 0.0000000001
    lvar = location_variance(g, lat_col, lng_col)
    if lvar is None:
        return None
    elif lvar>tiny_val:
        return np.log10(lvar)
    elif lvar <= tiny_val:
        return np.log10(tiny_val)
    else:
        return None

def number_of_clusters(g, loc_label_col="location_label"):
    if g is None or len(g) == 0:
        return None
    g =  g.dropna(how='any') # should not be required if input is static
    g = g.drop(g[(g.location_label < 1)].index)  # remove outliers/ cluster noise
    uniquelst = g[loc_label_col].unique()
    return len(uniquelst)

def moving_time_percent(g, loc_label_col="location_label"):
    if g is None or len(g) == 0:
        return None
    lbls = g[loc_label_col]
    nummoving = lbls.isnull().sum()
    numtotal = len(lbls)
    # print (nummoving)
    # print(numtotal)
    return (float(nummoving)/numtotal)

def outliers_time_percent(g, loc_label_col="location_label"):
    if g is None or len(g) == 0:
        return None
    lbls = g[loc_label_col]
    numoutliers = lbls[(lbls == -1)].sum()
    numoutliers = -1*numoutliers
    numtotal = len(lbls)
    return (float(numoutliers) / numtotal)


