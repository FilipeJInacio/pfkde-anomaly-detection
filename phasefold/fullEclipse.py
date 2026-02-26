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

def TLE_to_pos_vel(TLE1, TLE2, jd, jf):
    # Create a satellite object

    satellite = Satrec.twoline2rv(TLE1, TLE2)

    e, r, v = satellite.sgp4(jd, jf)

    # Create a CartesianDifferential object with the velocity
    velocity_diff = CartesianDifferential(*v, unit=u.km/u.s)

    # Create a CartesianRepresentation object with the position and velocity
    cart_rep = CartesianRepresentation(*r, unit=u.km, differentials=velocity_diff)

    # Create a TEME coordinate object with the CartesianRepresentation
    teme_p = TEME(cart_rep, obstime=Time(jd+jf, format='jd'))

    # Convert the TEME coordinates to ITRS
    itrf_p = teme_p.transform_to(ITRS(obstime=Time(jd+jf, format='jd')))

    return itrf_p.cartesian.xyz.to(u.km), itrf_p.velocity.d_xyz.to(u.km/u.s)

# field names
fields = "Start Time [UTC], End Time [UTC]\n"

# name of csv file
csvFile = open("eclipses_40014.csv", 'w')
csvFile.write(fields)

# Open the JSON file
f = open('40014_TLE.json')

# returns JSON object as a dictionary
jsonData = json.load(f)

# Load the trajectory configuration
with open(r"trajectory_bugsat.yml") as fp:
    yaml = YAML()
    yamldata = yaml.load(fp)


def set_3d_axes_scale_equal(ax: plt.Axes) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1]-x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1]-y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1]-z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*np.max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Earth Radius [km]
occultRadius = 6378
# sun radius [km]
sourceRadius = 696340.0

for i, line in enumerate(jsonData):
    print("Processing line", i)
    e0 = tempo.Epoch(line["EPOCH"]+" UTC")
    # Check if this is not the last line
    if i < len(jsonData) - 1:
        # Access the next line
        ef = tempo.Epoch(jsonData[i+1]["EPOCH"]+" UTC")
    else:
        break
    # Compute the position and velocity of the satellite at the initial epoch
    daysAndFraction = e0.jdPair(tempo.TimeScale.TT, tempo.JulianDay.JD)
    pos_vel = TLE_to_pos_vel(line["TLE_LINE1"], line["TLE_LINE2"], daysAndFraction.day, daysAndFraction.fraction)

    yamldata['timeline'][0]['epoch'] = e0.calStr("UTC", 6)
    yamldata['timeline'][0]['state'][0]['value']['pos_x'] = str(pos_vel[0][0])
    yamldata['timeline'][0]['state'][0]['value']['pos_y'] = str(pos_vel[0][1])
    yamldata['timeline'][0]['state'][0]['value']['pos_z'] = str(pos_vel[0][2])
    yamldata['timeline'][0]['state'][0]['value']['vel_x'] = str(str(pos_vel[1][0].value) + " km/s")
    yamldata['timeline'][0]['state'][0]['value']['vel_y'] = str(str(pos_vel[1][1].value) + " km/s")
    yamldata['timeline'][0]['state'][0]['value']['vel_z'] = str(str(pos_vel[1][2].value) + " km/s")
    yamldata['timeline'][1]['point']['epoch'] = (ef+1).calStr("UTC", 6)

    # Save the trajectory configuration
    with open(r"trajectory_bugsat.yml", "w") as out:
        yaml.dump(yamldata, out)

    # Load the universe configuration and create the universe object
    uniConfig = cosmos.util.load_yaml('universe_bugsat.yml')
    uni = cosmos.Universe(uniConfig)

    # Load the trajectory configuration and create the trajectory object using the universe object
    traConfig = cosmos.util.load_yaml('trajectory_bugsat.yml')
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
            csvFile.write(str(grid[e].calStr("UTC", 6)) + ", ")
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