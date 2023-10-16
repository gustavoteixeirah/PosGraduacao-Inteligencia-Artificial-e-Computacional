import folium

# Latitude and longitude coordinates for each city
city_coordinates = {
    'Maringá, PR': (-23.4273, -51.9375),
    'Londrina, PR': (-23.3103, -51.1629),
    'Cascavel, PR': (-24.9555, -53.4554),
    'Toledo, PR': (-24.7241, -53.7437),
    'Guarapuava, PR': (-25.3969, -51.458),
    'Curitiba, PR': (-25.4284, -49.2733),
    'Paranaguá, PR': (-25.5206, -48.5097),
    'Joinville, SC': (-26.3045, -48.8487),
    'Matinhos, PR': (-25.8209, -48.5357),
    'Jandaia do Sul, PR': (-23.6016, -51.6444),
    'Palotina, PR': (-24.2846, -53.8381)
}

# Centering the map on Santo Ângelo, RS
santo_angelo_coords = (-28.3003, -54.2666)
map_santo_angelo = folium.Map(location=santo_angelo_coords, zoom_start=6)

# Adding markers for each city
for city, coords in city_coordinates.items():
    folium.Marker(coords, popup=city).add_to(map_santo_angelo)

# Save the map to an HTML file
map_santo_angelo.show_in_browser()
# map_santo_angelo.save('cities_map.html')
import folium

# Latitude and longitude coordinates for each city
city_coordinates = {
    'Start': (-28.3003, -54.2666),  # Santo Ângelo, RS
    'End': (-24.9555, -53.4554)  # Cascavel, PR
}

# Create a map centered on the starting point
map_route = folium.Map(location=city_coordinates['Start'], zoom_start=6)

# Add markers for the start and end points
for city, coords in city_coordinates.items():
    folium.Marker(coords, popup=city).add_to(map_route)

# Create a line representing the route
route = [city_coordinates['Start'], city_coordinates['End']]
folium.PolyLine(route, color='blue', weight=2.5, opacity=1).add_to(map_route)

# Display the map with the route
map_route.show_in_browser()
