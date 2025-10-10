# Drone path planning algorithm
import numpy as np

def drone_path(start, end, charging_stations):
    """
    Generate a path for drone from start to end through charging stations.
    Currently returns a simple path visiting 3 random charging stations.

    Args:
        start (np.array): Starting position [x, y, z]
        end (np.array): Ending position [x, y, z]
        charging_stations (np.array): Array of charging station positions

    Returns:
        list: List of waypoints as np.array
    """
    path = [start.copy()]
    current = start.copy()

    # Select 3 random charging stations to visit
    if len(charging_stations) >= 3:
        stations_to_visit = np.random.choice(len(charging_stations), size=3, replace=False)
        for idx in stations_to_visit:
            station = charging_stations[idx]
            path.append(station.copy())
            current = station

    path.append(end.copy())
    return path
