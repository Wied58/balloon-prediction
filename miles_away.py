from geopy.distance import geodesic

# Coordinates in decimal degrees
point1 = (41.4343353333, -86.8928591667)
point2 = (40.8843675, -86.873666)

# Calculate the distance
distance = geodesic(point1, point2).miles
print(f"Distance: {distance} miles")
