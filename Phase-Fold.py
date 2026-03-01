import json
from godot.core import tempo
from godot import cosmos
import godot.core.astro as astro
from ruamel.yaml import YAML
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import TEME, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time
from sgp4.api import Satrec
import logging

# Earth Radius [km]
occultRadius = 6378
# sun radius [km]
sourceRadius = 696340.0

logger = logging.getLogger()
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.FileHandler("data/log.txt", mode='a')
handler.setFormatter(formatter)
logger.addHandler(handler)

def TLE_to_pos_vel(TLE1, TLE2, jd, jf):
    satellite = Satrec.twoline2rv(TLE1, TLE2) # Give keplerian elements to create the satellite object
    e, r, v = satellite.sgp4(jd, jf)

    if e != 0:
        logger.error(f"Error in SGP4 propagation: {e}\n TLE1: {TLE1}\n TLE2: {TLE2}\n JD: {jd}\n JF: {jf}")

    velocity_diff = CartesianDifferential(*v, unit=u.km/u.s)
    cart_rep = CartesianRepresentation(*r, unit=u.km, differentials=velocity_diff)
    teme_p = TEME(cart_rep, obstime=Time(jd+jf, format='jd'))
    itrf_p = teme_p.transform_to(ITRS(obstime=Time(jd+jf, format='jd')))

    return itrf_p.cartesian.xyz.to(u.km), itrf_p.velocity.d_xyz.to(u.km/u.s) # change to International Terrestrial Reference System

# CSV where the data will be saved
csvFile = open("data/eclipses_40014.csv", 'w')
csvFile.write("Start Time [UTC], End Time [UTC]\n")

# Open the JSON file with the TLE data
f = open('data/40014_TLE.json')
jsonData = json.load(f)

# Load the GODOT trajectory configuration
with open(r"data/trajectory_bugsat.yml") as fp:
    yaml = YAML()
    yamldata = yaml.load(fp)

for i, (current, nxt) in enumerate(zip(jsonData, jsonData[1:])):
    e0 = tempo.Epoch(current["EPOCH"] + " UTC") # Auto converts to TAI
    e0_JD = e0.jdPair(tempo.TimeScale.TT, tempo.JulianDay.JD) # Convert to Julian Date
    ef = tempo.Epoch(nxt["EPOCH"] + " UTC")

    # Compute the position and velocity of the satellite at the initial epoch
    pos_vel = TLE_to_pos_vel(current["TLE_LINE1"], current["TLE_LINE2"], e0_JD.day, e0_JD.fraction)

    # Configure GODOT trajectory
    yamldata['timeline'][0]['epoch'] = e0.calStr("UTC", 6)
    yamldata['timeline'][0]['state'][0]['value']['pos_x'] = str(pos_vel[0][0])
    yamldata['timeline'][0]['state'][0]['value']['pos_y'] = str(pos_vel[0][1])
    yamldata['timeline'][0]['state'][0]['value']['pos_z'] = str(pos_vel[0][2])
    yamldata['timeline'][0]['state'][0]['value']['vel_x'] = str(str(pos_vel[1][0].value) + " km/s")
    yamldata['timeline'][0]['state'][0]['value']['vel_y'] = str(str(pos_vel[1][1].value) + " km/s")
    yamldata['timeline'][0]['state'][0]['value']['vel_z'] = str(str(pos_vel[1][2].value) + " km/s")
    yamldata['timeline'][1]['point']['epoch'] = (ef+1).calStr("UTC", 6)

    # Save the trajectory configuration
    with open("data/trajectory_bugsat.yml", "w") as out:
        yaml.dump(yamldata, out)

    uniConfig = cosmos.util.load_yaml('data/universe_bugsat.yml')
    uni = cosmos.Universe(uniConfig)
    traConfig = cosmos.util.load_yaml('data/trajectory_bugsat.yml')
    tra = cosmos.Trajectory(uni, traConfig)

    # The trajectory evaluation: in this phase the timeline elements are processed and propagation arcs are computed
    tra.compute(partials = False)

    # Create a time grid with interval every 1 second
    grid = list(tempo.EpochRange(e0, ef).createGrid(1))

    # Retrieve propagated states from frames
    fra = uni.frames

    # Create a list to store the eclipses
    eclipses = []

    # Compute the occultation coefficient and the occultation margin for each epoch in the grid
    for e in grid:
        occultationCoefficient = astro.computeOccultationCoefficient(fra.vector3('SC_center', 'Earth','ICRF', e), fra.vector3('SC_center', 'Sun','ICRF', e), occultRadius, sourceRadius)
        occultationMargin = astro.computeOccultationMargin(occultationCoefficient)
        eclipses.append(occultationMargin.total)

    # Check for eclipses
    e = 0
    while e < len(eclipses):
        if eclipses[e] < 0:

            start = grid[e]
            csvFile.write(str(grid[e].calStr("UTC", 6)) + ",")
            while e < len(eclipses) and eclipses[e] < 0:
                e += 1
            if e < len(eclipses):  # Check if e is within the range after increment
                end = grid[e]
                csvFile.write(str(grid[e].calStr("UTC", 6)) + "\n")
            else:
                # Handle the case where the eclipse continues until the end of the grid
                # Optionally, write the last known end time or a custom message
                csvFile.write(ef.calStr("UTC", 6) + "\n")
                break  
        e += 1
    

# Closing file
f.close()
csvFile.close()