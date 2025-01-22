# Battery Swapping Simulation for Electric Vehicles in Metro Manila
# This program simulates the operation of an electric vehicle battery swapping network,
# tracking vehicle movements, battery states, and station operations.

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import folium
import random
from shapely.geometry import Point
from streamlit_folium import folium_static
import time
from collections import deque
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import os
import math

# Configuration Constants
max_history_length = 1000000  # Reduced from 5000
sample_rate = 1  # Only store data every N timesteps
TIME_STEP_MINUTES = 15  # Each simulation step represents 5 minutes
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
VISUALIZATION_UPDATE_INTERVAL = 2  # Update visualization every N timesteps
EMERGENCY_SWAP_HOURS = 2
TIMESTEPS_FOR_EMERGENCY = int((EMERGENCY_SWAP_HOURS * MINUTES_PER_HOUR) / TIME_STEP_MINUTES)
PATH_UPDATE_INTERVAL = 5  # Update paths every 5 time steps

# Add NCR Cities and their coordinates
NCR_CITIES = {
    'Manila': (14.5995, 120.9842),
    'Quezon City': (14.6760, 121.0437),
    'Makati': (14.5547, 121.0244),
    'Taguig': (14.5176, 121.0509),
    'Pasig': (14.5764, 121.0851),
    'Mandaluyong': (14.5794, 121.0359),
    'Pasay': (14.5378, 121.0014),
    'Parañaque': (14.4793, 120.9836),
    'Caloocan': (14.6497, 120.9784),
    'Marikina': (14.6507, 121.1029),
    'Muntinlupa': (14.4081, 121.0415),
    'Las Piñas': (14.4499, 120.9836),
    'Valenzuela': (14.7011, 120.9830),
    'Navotas': (14.6667, 120.9436),
    'Malabon': (14.6692, 120.9569),
    'San Juan': (14.6019, 121.0355),
    'Pateros': (14.5456, 121.0689)
}

# Add new configuration constants
SIMULATION_START_HOUR = 8  # 8 AM
SIMULATION_END_HOUR = 21   # 9 PM
HOURS_PER_CYCLE = 13      # 9 PM - 8 AM
TIMESTEPS_PER_HOUR = int(MINUTES_PER_HOUR / TIME_STEP_MINUTES)
TIMESTEPS_PER_CYCLE = HOURS_PER_CYCLE * TIMESTEPS_PER_HOUR

# Add new constants for battery specifications
BATTERY_UNIT_CAPACITY = 1.0  # kWh per battery unit
BATTERY_VOLTAGE = 48  # Volts

# Define Vehicle Categories
class VehicleCategory:
    """Defines different types of electric vehicles with specific characteristics
    
    Attributes:
        name: Category identifier (e.g., 'Type A', 'Type B')
        battery_count: Number of battery units required
        daily_range: Expected daily travel distance in km
        dod_threshold: Depth of Discharge threshold as percentage before requiring swap
        consumption_rate: Energy consumption in kWh per km
    """
    def __init__(self, name, battery_count, daily_range, dod_threshold, consumption_rate):
        self.name = name
        self.battery_count = battery_count  # number of battery units
        self.battery_capacity = battery_count * BATTERY_UNIT_CAPACITY  # total capacity in kWh
        self.daily_range = daily_range  # in km
        self.dod_threshold = dod_threshold  # Depth of Discharge threshold (%)
        self.consumption_rate = consumption_rate  # kWh per km

# Define Station Categories
class StationCategory:
    """Defines different types of battery swapping stations
    
    Attributes:
        name: Station type identifier (e.g., 'Small', 'Medium', 'Large')
        charging_time: Time needed to charge a battery (in hours)
        inventory_capacity: Maximum number of batteries that can be stored
        charging_slots: Number of simultaneous charging positions
        battery_types: List of vehicle categories this station can service
    """
    def __init__(self, name, charging_time, inventory_capacity, charging_slots, battery_types=None):
        self.name = name
        self.charging_time = charging_time  # in hours
        self.inventory_capacity = inventory_capacity
        self.charging_slots = charging_slots
        self.battery_types = battery_types or []

# Vehicle Class
class Vehicle:
    """Represents an individual electric vehicle in the simulation
    
    Handles vehicle movement, battery consumption, and swap requests.
    Tracks position, state of charge, and route information.
    """
    def __init__(self, vehicle_id, category, initial_soc, route, home_coordinate):
        self.vehicle_id = vehicle_id
        self.category = category
        # initial_soc is now in terms of battery energy (kWh)
        self.soc = min(initial_soc, category.battery_capacity)  
        self.route = route
        self.route_index = 0
        self.position = route[0]
        self.needs_swap = False
        self.distance_traveled = 0
        self.out_of_charge = False
        self.out_of_charge_duration = 0
        self.emergency_swaps = 0
        self.last_path_update = 0
        self.route_lock = threading.Lock()
        self.home_coordinate = home_coordinate
        self.initial_route = route.copy()
        self.regular_swaps = 0
        
    def get_battery_count(self):
        """Returns the current number of functional battery units"""
        return int(self.soc / BATTERY_UNIT_CAPACITY)
    
    def get_soc_percentage(self):
        """Returns the current state of charge as a percentage"""
        return (self.soc / self.category.battery_capacity) * 100

    def get_soc_color(self):
        """Return a color based on SOC percentage"""
        soc_percentage = self.get_soc_percentage()
        if soc_percentage > 75:
            return 'green'
        elif soc_percentage > 50:
            return 'yellowgreen'
        elif soc_percentage > 25:
            return 'orange'
        else:
            return 'red'

    def check_emergency_swap(self, G, stations, current_timestep):
        """Check if vehicle is eligible for emergency swap and perform it if possible"""
        if self.out_of_charge:
            self.out_of_charge_duration += 1
            if self.out_of_charge_duration >= TIMESTEPS_FOR_EMERGENCY:
                nearest_stations = find_nearest_station(G, self.position, stations)
                # Try each station until we find one that works
                for station in nearest_stations:
                    if station.is_operating:
                        if station.accept_battery(self, current_timestep):
                            self.soc = self.category.battery_capacity
                            self.out_of_charge = False
                            self.needs_swap = False
                            self.out_of_charge_duration = 0
                            self.emergency_swaps += 1
                            return True
        return False

    def move(self, G, stations, current_timestep):
        """Updates vehicle position and energy consumption based on movement"""
        # First check for emergency swap
        if self.check_emergency_swap(G, stations, current_timestep):
            return

        if self.out_of_charge:
            return

        # Check if path needs updating
        if self.should_update_path(current_timestep):
            new_route = self.calculate_new_path(G, stations)
            # Verify route ends at a station if swap needed
            if self.needs_swap and stations:
                destination = new_route[-1]
                ends_at_station = any(station.location == destination for station in stations)
                if not ends_at_station:
                    # Force recalculation to ensure path ends at station
                    nearest_stations = find_nearest_station(G, self.position, stations)
                    if nearest_stations:
                        try:
                            new_route = nx.shortest_path(G, self.position, nearest_stations[0].location, weight='length')
                        except nx.NetworkXNoPath:
                            pass
            
            self.update_route(G, new_route)
            self.last_path_update = current_timestep

        if self.route_index < len(self.route) - 1:
            prev_position = self.position
            self.route_index += 1
            self.position = self.route[self.route_index]

            edge_data = G.get_edge_data(prev_position, self.position)
            if edge_data:
                distance = edge_data[0]['length'] / 1000  # Convert meters to km
                
                # Calculate actual distance covered in this time step
                average_speed = 30  # km/h
                time_fraction = TIME_STEP_MINUTES / MINUTES_PER_HOUR  # convert minutes to hours
                actual_distance = min(distance, average_speed * time_fraction)
                
                self.distance_traveled += actual_distance
                energy_consumed = actual_distance * self.category.consumption_rate
                self.soc -= energy_consumed

                if self.soc <= self.category.battery_capacity * self.category.dod_threshold:
                    self.needs_swap = True
                    # Immediately calculate path to nearest station when swap needed
                    new_route = self.calculate_new_path(G, stations)
                    self.update_route(G, new_route)
                    self.last_path_update = current_timestep
                if self.soc <= 0:
                    self.out_of_charge = True
                    self.soc = 0
        else:
            self.route = generate_random_route(G)
            self.route_index = 0

    def swap_battery(self, station, current_timestep):
        """Track successful battery swaps"""
        if not self.out_of_charge and station.accept_battery(self, current_timestep):
            self.soc = self.category.battery_capacity
            self.needs_swap = False
            self.regular_swaps += 1
            return True
        return False

    def update_route(self, G, new_route):
        """Thread-safe route update"""
        with self.route_lock:
            self.route = new_route
            self.route_index = 0

    def should_update_path(self, current_timestep):
        """Check if vehicle needs path update"""
        return (
            self.last_path_update + PATH_UPDATE_INTERVAL <= current_timestep or  # Every 5 steps
            self.needs_swap or  # When heading to station
            self.route_index >= len(self.route) - 1  # When route completed
        )

    def calculate_new_path(self, G, stations=None):
        """Calculate new path based on current situation"""
        if self.needs_swap and not self.out_of_charge and stations:
            # Get sorted list of nearest stations
            nearest_stations = find_nearest_station(G, self.position, stations)
            
            if not nearest_stations:
                return generate_random_route(G)  # No available stations
                
            # Try each station until we find a valid path
            for station in nearest_stations:
                try:
                    path = nx.shortest_path(G, self.position, station.location, weight='length')
                    return path
                except nx.NetworkXNoPath:
                    continue
                    
            # If no path found to any station, generate random route
            return generate_random_route(G)
        
        # If no swap needed, first generate random route
        route = generate_random_route(G)
        
        # If vehicle needs swap, verify destination is a station
        if self.needs_swap and stations:
            # Find nearest station to the random destination
            nearest_stations = find_nearest_station(G, route[-1], stations)
            if nearest_stations:
                try:
                    # Create path: current -> random destination -> nearest station
                    path_to_station = nx.shortest_path(G, route[-1], nearest_stations[0].location, weight='length')
                    # Combine paths, removing duplicate connection point
                    return route[:-1] + path_to_station
                except nx.NetworkXNoPath:
                    pass
            
            # If path to station fails, try direct path to nearest station from current position
            nearest_stations = find_nearest_station(G, self.position, stations)
            if nearest_stations:
                try:
                    return nx.shortest_path(G, self.position, nearest_stations[0].location, weight='length')
                except nx.NetworkXNoPath:
                    pass
                    
        return route

    def reset_for_new_day(self):
        """Reset vehicle for new day at 8AM"""
        self.soc = self.category.battery_capacity  # Full charge
        self.position = self.home_coordinate  # Return to home
        self.needs_swap = False
        self.out_of_charge = False
        self.out_of_charge_duration = 0
        self.route = self.initial_route.copy()  # Reset to initial route
        self.route_index = 0
        self.last_path_update = 0

# Station Class
class Station:
    """Represents a battery swapping station
    
    Manages battery inventory, charging operations, and swap requests.
    """
    def __init__(self, station_id, category, location):
        self.station_id = station_id
        self.category = category
        self.location = location
        self.inventory = category.inventory_capacity  # Start with all batteries charged
        self.charging_slots = category.charging_slots
        self.charging_batteries = []  # List of (timestep_started, charging_duration) tuples
        self.is_operating = True
        self.depleted_inventory = 0  # Track depleted batteries waiting to be charged
        self.total_battery_count = category.inventory_capacity  # Track total batteries at station

    def accept_battery(self, vehicle, current_timestep):
        """Modified to handle battery swaps correctly"""
        if not self.is_operating:
            return False
            
        # Check if we have charged batteries to give
        if self.inventory <= 0:
            return False
        
        # For emergency swaps
        if vehicle.out_of_charge and vehicle.out_of_charge_duration >= TIMESTEPS_FOR_EMERGENCY:
            # Give charged battery from inventory
            self.inventory -= 1
            # Add depleted battery to station
            if len(self.charging_batteries) < self.charging_slots:
                charging_duration = int((self.category.charging_time * MINUTES_PER_HOUR) / TIME_STEP_MINUTES)
                self.charging_batteries.append((current_timestep, charging_duration))
            else:
                self.depleted_inventory += 1
            return True
            
        # For regular swaps - check vehicle compatibility
        if vehicle.category.name not in self.category.battery_types:
            return False
            
        # Give charged battery and handle depleted one
        self.inventory -= 1
        if len(self.charging_batteries) < self.charging_slots:
            charging_duration = int((self.category.charging_time * MINUTES_PER_HOUR) / TIME_STEP_MINUTES)
            self.charging_batteries.append((current_timestep, charging_duration))
        else:
            self.depleted_inventory += 1
        return True

    def charge_batteries(self, current_timestep):
        """Modified to handle completed charging and waiting batteries"""
        # Check which batteries are done charging
        completed_batteries = []
        remaining_batteries = []
        
        for start_time, duration in self.charging_batteries:
            if current_timestep - start_time >= duration:
                completed_batteries.append((start_time, duration))
            else:
                remaining_batteries.append((start_time, duration))
        
        # Add completed batteries to inventory
        num_completed = len(completed_batteries)
        self.inventory += num_completed
        
        # Start charging waiting batteries if slots became available
        slots_freed = num_completed
        while slots_freed > 0 and self.depleted_inventory > 0:
            charging_duration = int((self.category.charging_time * MINUTES_PER_HOUR) / TIME_STEP_MINUTES)
            remaining_batteries.append((current_timestep, charging_duration))
            self.depleted_inventory -= 1
            slots_freed -= 1
        
        # Keep track of batteries still charging
        self.charging_batteries = remaining_batteries

    def reset_for_new_day(self):
        """Reset station state for new day"""
        # Calculate total batteries at the station
        total_batteries = (self.inventory + self.depleted_inventory + 
                         len(self.charging_batteries))
        
        # Log warning if total doesn't match expected
        if total_batteries != self.total_battery_count:
            print(f"Warning: Station {self.station_id} battery count mismatch. " 
                  f"Expected {self.total_battery_count}, found {total_batteries}")
        
        # Reset all batteries to charged state
        self.inventory = self.total_battery_count
        self.depleted_inventory = 0
        self.charging_batteries.clear()  # Explicitly clear the charging slots
        self.is_operating = False  # Ensure station is closed during non-operating hours
        
        # Double-check everything is reset properly
        assert self.inventory == self.total_battery_count, (
            f"Station {self.station_id} inventory not reset properly")
        assert self.depleted_inventory == 0, (
            f"Station {self.station_id} depleted inventory not cleared")
        assert len(self.charging_batteries) == 0, (
            f"Station {self.station_id} charging slots not cleared")

    def update_operating_status(self, current_hour):
        """Update station operating status based on time of day"""
        was_operating = self.is_operating
        self.is_operating = SIMULATION_START_HOUR <= current_hour < SIMULATION_END_HOUR
        
        # If transitioning from operating to closed, reset the station
        if was_operating and not self.is_operating:
            self.reset_for_new_day()

# Helper functions
def generate_random_route(G):
    """Generates a random route between two points in the network
    
    Args:
        G: NetworkX graph representing the road network
    Returns:
        List of node IDs representing the route
    """
    origin = random.choice(list(G.nodes))
    destination = random.choice(list(G.nodes))
    try:
        route = nx.shortest_path(G, origin, destination, weight='length')
        return route
    except nx.NetworkXNoPath:
        return generate_random_route(G)

def find_nearest_station(G, vehicle_position, stations):
    """Modified to return all stations sorted by distance"""
    available_stations = []
    for station in stations:
        if not station.is_operating or station.inventory <= 0:
            continue  # Skip closed stations or those without inventory
        try:
            distance = nx.shortest_path_length(G, vehicle_position, station.location, weight='length')
            available_stations.append((station, distance))
        except nx.NetworkXNoPath:
            continue
    
    # Sort stations by distance
    available_stations.sort(key=lambda x: x[1])
    return [station for station, _ in available_stations]

def create_map(G, vehicles, stations):
    """Creates a Folium map visualization of the current simulation state"""
    # Get center coordinates and create map
    center_node = list(G.nodes)[0]
    center_lat = G.nodes[center_node]['y']
    center_lon = G.nodes[center_node]['x']
    
    bounds = ox.graph_to_gdfs(G, nodes=True, edges=False).total_bounds
    radius_meters = geodesic((bounds[1], bounds[0]), (bounds[3], bounds[2])).meters

    # Create map with light style
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=get_zoom_level(radius_meters),
        tiles='cartodbpositron'  # Light map style
    )

    # Add stations with lighter blue
    for station in stations:
        lon, lat = G.nodes[station.location]['x'], G.nodes[station.location]['y']
        folium.CircleMarker(
            location=(lat, lon),
            radius=8,
            color='darkblue',
            fill=True,
            fill_color='darkblue',
            fill_opacity=0.9,
            popup=f"Station ID: {station.station_id}\nInventory: {station.inventory}"
        ).add_to(m)

    # Add vehicles with directional triangles, tails, and route paths
    for vehicle in vehicles:
        current_pos = vehicle.position
        current_lon, current_lat = G.nodes[current_pos]['x'], G.nodes[current_pos]['y']
        
        # Calculate direction if there's a next position
        rotation_angle = 0
        if vehicle.route_index < len(vehicle.route) - 1:
            next_pos = vehicle.route[vehicle.route_index + 1]
            next_lon, next_lat = G.nodes[next_pos]['x'], G.nodes[next_pos]['y']
            # Calculate angle between current and next position
            dx = next_lon - current_lon
            dy = next_lat - current_lat
            rotation_angle = math.degrees(math.atan2(dy, dx))
            
            # Draw remaining route path only if vehicle needs swap
            if vehicle.needs_swap and vehicle.route_index < len(vehicle.route) - 1:
                # Get coordinates for remaining route
                route_coords = []
                for node in vehicle.route[vehicle.route_index:]:
                    node_lat = G.nodes[node]['y']
                    node_lon = G.nodes[node]['x']
                    route_coords.append((node_lat, node_lon))
                
                # Draw dashed line along the route
                folium.PolyLine(
                    locations=route_coords,
                    color=vehicle.get_soc_color(),
                    weight=2,
                    opacity=0.5,
                    dash_array='5, 10',  # Creates dashed line pattern
                    popup=f"Vehicle {vehicle.vehicle_id}'s route to station"
                ).add_to(m)
                
                # Add small dot at destination
                dest_lat = G.nodes[vehicle.route[-1]]['y']
                dest_lon = G.nodes[vehicle.route[-1]]['x']
                folium.CircleMarker(
                    location=(dest_lat, dest_lon),
                    radius=3,
                    color=vehicle.get_soc_color(),
                    fill=True,
                    fill_color=vehicle.get_soc_color(),
                    fill_opacity=0.5,
                    popup=f"Destination for Vehicle {vehicle.vehicle_id}"
                ).add_to(m)
        
        popup_text = f"""
        Vehicle ID: {vehicle.vehicle_id}
        Category: {vehicle.category.name}
        Battery Units: {vehicle.get_battery_count()}/{vehicle.category.battery_count}
        SOC: {vehicle.get_soc_percentage():.1f}%
        Needs Swap: {'Yes' if vehicle.needs_swap else 'No'}
        Out of Charge: {'Yes' if vehicle.out_of_charge else 'No'}
        """
        
        # Create arrow icon (triangle with tail)
        arrow_icon = folium.DivIcon(
            html=f'''
                <div style="
                    position: relative;
                    transform: rotate({rotation_angle}deg);
                ">
                    <!-- Tail (rectangle) -->
                    <div style="
                        position: absolute;
                        width: 8px;
                        height: 4px;
                        background-color: {vehicle.get_soc_color()};
                        left: -4px;
                        top: 4px;
                    "></div>
                    <!-- Head (triangle) -->
                    <div style="
                        position: absolute;
                        width: 0;
                        height: 0;
                        border-left: 6px solid transparent;
                        border-right: 6px solid transparent;
                        border-bottom: 12px solid {vehicle.get_soc_color()};
                        left: -6px;
                        top: -8px;
                    "></div>
                </div>
            ''',
            icon_size=(16, 16),
            icon_anchor=(8, 8),
        )
        
        folium.Marker(
            location=(current_lat, current_lon),
            icon=arrow_icon,
            popup=popup_text,
            tooltip=f"V{vehicle.vehicle_id}"
        ).add_to(m)

    return m

def load_stations_from_csv(uploaded_file, G, station_categories):
    """Load station data from CSV file
    
    Expected CSV format:
    station_id,category,latitude,longitude,initial_inventory
    1,Small,14.5995,120.9842,20
    
    Args:
        uploaded_file: Streamlit uploaded file object
        G: NetworkX graph
        station_categories: Dictionary of station categories
    Returns:
        List of Station objects
    """
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['station_id', 'category', 'latitude', 'longitude', 'initial_inventory']
        
        # Verify all required columns are present
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain columns: {', '.join(required_columns)}")
            return None
            
        stations = []
        for _, row in df.iterrows():
            # Find nearest node in graph to specified lat/lon
            nearest_node = ox.nearest_nodes(G, row['longitude'], row['latitude'])
            
            # Verify category exists
            if row['category'] not in station_categories:
                st.error(f"Invalid station category: {row['category']}")
                return None
                
            station = Station(
                station_id=row['station_id'],
                category=station_categories[row['category']],
                location=nearest_node
            )
            # Override default inventory if specified
            if 'initial_inventory' in row:
                station.inventory = min(row['initial_inventory'], 
                                     station.category.inventory_capacity)
            
            stations.append(station)
            
        return stations
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def update_vehicle_paths(vehicles, G, stations, current_timestep):
    """Update paths for all vehicles in parallel"""
    def update_single_vehicle(vehicle):
        if vehicle.should_update_path(current_timestep):
            new_route = vehicle.calculate_new_path(G, stations)
            vehicle.update_route(G, new_route)
            vehicle.last_path_update = current_timestep

    with ThreadPoolExecutor(max_workers=len(vehicles)) as executor:
        executor.map(lambda v: update_single_vehicle(v), vehicles)

def get_bounded_graph(center_city, radius_meters):
    """Get a graph bounded by center coordinates and radius
    
    Args:
        center_city: Name of NCR city to center on
        radius_meters: Radius in meters from center
    Returns:
        NetworkX graph of the bounded area
    """
    if center_city not in NCR_CITIES:
        raise ValueError(f"City {center_city} not found in NCR cities list")
    
    center_lat, center_lon = NCR_CITIES[center_city]
    
    # Get the graph within the specified radius
    G = ox.graph_from_point(
        center_point=(center_lat, center_lon),
        dist=radius_meters,
        network_type='drive',
        simplify=True
    )
    
    return G

def get_zoom_level(radius_meters):
    """Calculate appropriate zoom level based on radius
    
    Args:
        radius_meters: Radius in meters
    Returns:
        zoom_level: Integer zoom level for Folium map
    """
    # These values are calibrated for Metro Manila's scale
    if radius_meters <= 500:
        return 17
    elif radius_meters <= 1000:
        return 16
    elif radius_meters <= 2000:
        return 15
    elif radius_meters <= 3500:
        return 14
    else:
        return 13

# Add new helper functions
def get_current_hour(timestep):
    """Convert timestep to hour of day"""
    minutes_elapsed = timestep * TIME_STEP_MINUTES
    hour = (SIMULATION_START_HOUR + (minutes_elapsed // MINUTES_PER_HOUR)) % 24
    return hour

def is_end_of_cycle(timestep):
    """Check if current timestep is at the end of a cycle"""
    current_hour = get_current_hour(timestep)
    # Check if we're at the last timestep of the operating hours
    return current_hour >= SIMULATION_END_HOUR

def export_vehicle_data(results_df, filename="vehicle_data.csv"):
    """Export vehicle tracking data to CSV"""
    results_df.to_csv(filename, index=False)
    return filename

def export_station_data(station_history, filename="station_data.csv"):
    """Export station tracking data to CSV"""
    station_df = pd.DataFrame(station_history)
    station_df.to_csv(filename, index=False)
    return filename

def reset_simulation_for_new_day(vehicles, stations):
    """Reset all vehicles and stations for new day"""
    print(f"Resetting simulation for new day")  # Debug log
    
    # Reset all stations first
    for station in stations:
        station.reset_for_new_day()
    
    # Then reset all vehicles
    for vehicle in vehicles:
        vehicle.reset_for_new_day()

# Main Streamlit App
def main():
    """Main simulation application using Streamlit
    
    Handles:
    1. User interface and parameter input
    2. Simulation initialization
    3. Real-time simulation execution
    4. Visualization updates
    5. Results display
    """
    st.title("Battery Swapping Simulation in Metro Manila")

    # Add description and instructions
    with st.expander("📖 Description & Instructions", expanded=False):
        st.markdown("""
        ### Description
        This simulation models electric vehicle battery swapping operations in Metro Manila. It tracks vehicle movements, 
        battery states, and swapping station operations in real-time.

        ### How to Use
        1. **Area Settings** (Sidebar):
            - Select a center city in Metro Manila
            - Set the working radius (100-5000 meters)

        2. **Configure Vehicle Types** (Sidebar):
            - Expand 'Vehicle Categories Configuration'
            - Adjust parameters for each vehicle type (A, B, C)
            - Parameters include battery capacity, range, discharge threshold, and consumption rate

        3. **Configure Station Types** (Sidebar):
            - Expand 'Station Categories Configuration'
            - Adjust parameters for each station type (Small, Medium, Large)
            - Parameters include charging time, inventory capacity, and charging slots

        4. **Set Simulation Parameters** (Sidebar):
            - Set number of vehicles
            - Select vehicle categories to include
            - Adjust distribution percentage for each vehicle type

        5. **Run Simulation**:
            - Click 'Run Simulation' button
            - Monitor real-time metrics and charts:
                - Current out-of-charge vehicles
                - Emergency swap occurrences
                - Average battery levels by vehicle type
                - Daily statistics

        ### Map Legend
        - Vehicles are shown as colored triangles (green→red indicating battery level)
        - Stations are shown as blue markers
        - Hover over markers to see detailed information
        """)

    # Sidebar configurations
    st.sidebar.header("Area Settings")
    center_city = st.sidebar.selectbox(
        "Center City",
        options=list(NCR_CITIES.keys()),
        index=0
    )
    
    radius_meters = st.sidebar.slider(
        "Working Radius (meters)",
        min_value=100,
        max_value=5000,
        value=2000,
        step=100,
        help="Radius from city center where vehicles and stations will operate"
    )

    # Vehicle Categories Configuration
    with st.sidebar.expander("Vehicle Categories Configuration"):
        vehicle_categories = {}
        
        # Type A Config
        st.subheader("Type A (Light Vehicles)")
        type_a_batteries = st.slider("Number of Battery Units", 20, 80, 50, key="type_a_bat")
        type_a_range = st.slider("Daily Range (km)", 100, 300, 200, key="type_a_range")
        type_a_dod = st.slider("Depth of Discharge Threshold", 0.1, 0.6, 0.4, key="type_a_dod")
        type_a_consumption = st.slider("Consumption Rate (kWh/km)", 1.0, 10.0, 15.15, key="type_a_cons")
        vehicle_categories['Type A'] = VehicleCategory('Type A', type_a_batteries, type_a_range, type_a_dod, type_a_consumption)
        
        # Type B Config
        st.subheader("Type B (Medium Vehicles)")
        type_b_batteries = st.slider("Number of Battery Units", 40, 100, 75, key="type_b_bat")
        type_b_range = st.slider("Daily Range (km)", 200, 400, 300, key="type_b_range")
        type_b_dod = st.slider("Depth of Discharge Threshold", 0.1, 0.6, 0.4, key="type_b_dod")
        type_b_consumption = st.slider("Consumption Rate (kWh/km)", 5.0, 20.0, 25.25, key="type_b_cons")
        vehicle_categories['Type B'] = VehicleCategory('Type B', type_b_batteries, type_b_range, type_b_dod, type_b_consumption)
        
        # Type C Config
        st.subheader("Type C (Heavy Vehicles)")
        type_c_batteries = st.slider("Number of Battery Units", 60, 150, 100, key="type_c_bat")
        type_c_range = st.slider("Daily Range (km)", 300, 500, 400, key="type_c_range")
        type_c_dod = st.slider("Depth of Discharge Threshold", 0.1, 0.6, 0.4, key="type_c_dod")
        type_c_consumption = st.slider("Consumption Rate (kWh/km)", 10.0, 30.0, 35.35, key="type_c_cons")
        vehicle_categories['Type C'] = VehicleCategory('Type C', type_c_batteries, type_c_range, type_c_dod, type_c_consumption)

    # Station Categories Configuration
    with st.sidebar.expander("Station Categories Configuration"):
        station_categories = {}
        
        # Small Station Config
        st.subheader("Small Stations")
        small_charging_time = st.slider("Charging Time (hours)", 1, 6, 3, key="small_charging")
        small_inventory = st.slider("Inventory Capacity", 10, 30, 20, key="small_inventory")
        small_slots = st.slider("Charging Slots", 2, 8, 5, key="small_slots")
        station_categories['Small'] = StationCategory('Small', small_charging_time, small_inventory, small_slots, ['Type A'])
        
        # Medium Station Config
        st.subheader("Medium Stations")
        medium_charging_time = st.slider("Charging Time (hours)", 1, 6, 3, key="medium_charging")
        medium_inventory = st.slider("Inventory Capacity", 20, 60, 45, key="medium_inventory")
        medium_slots = st.slider("Charging Slots", 5, 15, 10, key="medium_slots")
        station_categories['Medium'] = StationCategory('Medium', medium_charging_time, medium_inventory, medium_slots, ['Type A', 'Type B'])
        
        # Large Station Config
        st.subheader("Large Stations")
        large_charging_time = st.slider("Charging Time (hours)", 1, 6, 3, key="large_charging")
        large_inventory = st.slider("Inventory Capacity", 40, 120, 90, key="large_inventory")
        large_slots = st.slider("Charging Slots", 10, 30, 20, key="large_slots")
        station_categories['Large'] = StationCategory('Large', large_charging_time, large_inventory, large_slots, ['Type A', 'Type B', 'Type C'])

    # Rest of sidebar inputs
    st.sidebar.header("Simulation Parameters")
    num_vehicles = st.sidebar.number_input("Number of Vehicles", min_value=1, value=100)
    
    selected_vehicle_categories = st.sidebar.multiselect(
        "Vehicle Categories",
        options=list(vehicle_categories.keys()),
        default=list(vehicle_categories.keys())
    )

    # Add vehicle category split sliders
    vehicle_category_split = {}
    with st.sidebar.expander("Vehicle Category Distribution"):
        remaining_percent = 100
        for category in selected_vehicle_categories[:-1]:  # All except last
            percent = st.slider(
                f"Percentage of {category}",
                min_value=0,
                max_value=remaining_percent,
                value=min(remaining_percent // len(selected_vehicle_categories), remaining_percent),
                key=f"vehicle_split_{category}"
            )
            vehicle_category_split[category] = percent
            remaining_percent -= percent
        # Last category gets the remainder
        if selected_vehicle_categories:
            last_category = selected_vehicle_categories[-1]
            vehicle_category_split[last_category] = remaining_percent
            st.info(f"Percentage of {last_category}: {remaining_percent}%")

    # Add file uploader before station parameters
    st.sidebar.header("Station Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Station Locations (CSV)", 
        type=['csv'], 
        help="CSV file with columns: station_id,category,latitude,longitude,initial_inventory"
    )
    
    # Show manual station configuration only if no file is uploaded
    if uploaded_file is None:
        num_stations = st.sidebar.number_input("Number of Swapping Stations", min_value=1, value=6)
        
        selected_station_categories = st.sidebar.multiselect(
            "Station Categories", 
            options=list(station_categories.keys()),
            default=list(station_categories.keys())
        )
        
        station_category_split = {}
        for category in selected_station_categories:
            percent = st.sidebar.slider(
                f"Percentage of {category} Stations", 
                min_value=0, 
                max_value=100, 
                value=100 // len(selected_station_categories)
            )
            station_category_split[category] = percent

    # Simulation Time
    simulation_time = st.sidebar.number_input("Simulation Time Steps", min_value=1, value=500)

    # Single Run button
    start_button = st.sidebar.button("Run Simulation")

    if start_button:
        # Initialize simulation
        with st.spinner('Initializing simulation...'):
            try:
                G = get_bounded_graph(center_city, radius_meters)
                st.info(f"Simulating area centered on {center_city} with {radius_meters}m radius")
                
                # Initialize data collection lists
                vehicle_history = []
                station_history = []  # Initialize station_history here
                
                # Initialize vehicles and stations
                vehicles = []
                vehicle_id = 0
                nodes = list(G.nodes)  # Get nodes within bounds
                
                for category_name, percent in vehicle_category_split.items():
                    num_category_vehicles = int((percent / 100) * num_vehicles)
                    for _ in range(num_category_vehicles):
                        # Ensure initial route is within bounds
                        route = generate_random_route(G)
                        max_capacity = vehicle_categories[category_name].battery_capacity
                        initial_soc = random.uniform(0.5 * max_capacity, max_capacity)
                        vehicle = Vehicle(
                            vehicle_id=vehicle_id,
                            category=vehicle_categories[category_name],
                            initial_soc=initial_soc,
                            route=route,
                            home_coordinate=random.choice(nodes)
                        )
                        vehicles.append(vehicle)
                        vehicle_id += 1

                # Initialize stations within bounds
                if uploaded_file is not None:
                    stations = load_stations_from_csv(uploaded_file, G, station_categories)
                    if stations is None:
                        st.error("Failed to load stations from CSV. Please check the file format.")
                        st.stop()
                else:
                    stations = []
                    station_id = 0
                    for category_name, percent in station_category_split.items():
                        num_category_stations = int((percent / 100) * num_stations)
                        for _ in range(num_category_stations):
                            # Select random node within bounds
                            location = random.choice(nodes)
                            station = Station(
                                station_id=station_id,
                                category=station_categories[category_name],
                                location=location
                            )
                            stations.append(station)
                            station_id += 1
                            
            except Exception as e:
                st.error(f"Error initializing simulation: {str(e)}")
                st.stop()

        # Create placeholders for live updates
        map_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Initialize results storage with deque
        results = deque(maxlen=max_history_length)
        
        # Simulation Loop
        progress_bar = st.progress(0)
        cycle = 0
        while cycle * TIMESTEPS_PER_CYCLE < simulation_time:
            for t in range(TIMESTEPS_PER_CYCLE):
                current_timestep = cycle * TIMESTEPS_PER_CYCLE + t
                current_hour = get_current_hour(t)
                
                # Check for start of new day (8AM)
                if current_hour == SIMULATION_START_HOUR:
                    reset_simulation_for_new_day(vehicles, stations)
                
                # Update station operating status
                for station in stations:
                    station.update_operating_status(current_hour)
                
                # Update paths in parallel
                update_vehicle_paths(vehicles, G, stations, current_timestep)
                
                # Move vehicles and handle battery swaps
                for vehicle in vehicles:
                    vehicle.move(G, stations, current_timestep)
                    
                    # Record vehicle state
                    results.append({
                        'timestep': current_timestep,
                        'hour': current_hour,
                        'vehicle_id': vehicle.vehicle_id,
                        'category': vehicle.category.name,
                        'position': vehicle.position,
                        'soc': vehicle.soc,
                        'needs_swap': vehicle.needs_swap,
                        'out_of_charge': vehicle.out_of_charge,
                        'emergency_swap': vehicle.emergency_swaps > 0
                    })
                
                # Record station state
                for station in stations:
                    station.charge_batteries(current_timestep)
                    station_history.append({
                        'timestep': current_timestep,
                        'hour': current_hour,
                        'station_id': station.station_id,
                        'category': station.category.name,
                        'inventory': station.inventory,
                        'charging_slots_used': len(station.charging_batteries),
                        'is_operating': station.is_operating
                    })
                
                # Calculate metrics
                num_out_of_charge = sum(1 for v in vehicles if v.out_of_charge)
                total_emergency_swaps = sum(v.emergency_swaps for v in vehicles)
                
                # Calculate average SOC per vehicle category
                category_soc = {}
                for category in selected_vehicle_categories:
                    category_vehicles = [v for v in vehicles if v.category.name == category]
                    if category_vehicles:
                        avg_soc = sum(v.soc for v in category_vehicles) / len(category_vehicles)
                        category_soc[category] = avg_soc

                # Convert results to DataFrame for visualization
                results_df = pd.DataFrame(results)
                
                # Calculate aggregate metrics if DataFrame is not empty
                if len(results_df) > 0:
                    # Update map visualization
                    if current_timestep % VISUALIZATION_UPDATE_INTERVAL == 0:
                        with map_placeholder.container():
                            m = create_map(G, vehicles, stations)
                            folium_static(m)
                    
                    # Calculate current metrics
                    num_out_of_charge = sum(1 for v in vehicles if v.out_of_charge)
                    total_emergency_swaps = sum(v.emergency_swaps for v in vehicles)
                    
                    # Update metrics visualizations
                    with metrics_placeholder.container():
                        col1, col2 = st.columns(2)  # Change to 2 columns
                        with col1:
                            st.metric("Current Vehicles Out of Charge", num_out_of_charge)
                        with col2:
                            # Combine regular and emergency swaps
                            total_swaps = sum(v.regular_swaps + v.emergency_swaps for v in vehicles)
                            st.metric("Total Battery Swaps", total_swaps)
                    
                    with chart_placeholder.container():
                        tab1, tab2, tab3, tab4 = st.tabs(["Vehicles Out of Charge", "Average SOC by Category", 
                                                         "Vehicle 0 SOC", "Station 0 Inventory"])
                        
                        with tab1:
                            # Track out-of-charge vehicles over time
                            out_of_charge_df = results_df.groupby('timestep')['out_of_charge'].sum().reset_index()
                            chart_data = pd.DataFrame(index=range(simulation_time))
                            chart_data['timestep'] = range(simulation_time)
                            chart_data['out_of_charge'] = 0
                            chart_data.update(out_of_charge_df.set_index('timestep'))
                            
                            fig = px.line(chart_data, x='timestep', y='out_of_charge', 
                                        title='Vehicles Out of Charge Over Time')
                            fig.update_layout(xaxis_range=[0, simulation_time])
                            st.plotly_chart(fig, use_container_width=True, key=f"out_of_charge_chart_{current_timestep}")
                            
                        with tab2:
                            # Calculate average SOC by category
                            category_soc_df = results_df.pivot_table(
                                index='timestep',
                                columns='category',
                                values='soc',
                                aggfunc='mean'
                            ).fillna(method='ffill')
                            
                            # Create fixed-size DataFrame with all timesteps
                            chart_data = pd.DataFrame(index=range(simulation_time))
                            chart_data['timestep'] = range(simulation_time)
                            for category in selected_vehicle_categories:
                                chart_data[f'Avg SOC {category}'] = category_soc_df.get(category, 0)
                            
                            fig = go.Figure()
                            for category in selected_vehicle_categories:
                                fig.add_trace(go.Scatter(
                                    x=chart_data['timestep'],
                                    y=chart_data[f'Avg SOC {category}'],
                                    name=f'Avg SOC {category}',
                                    mode='lines'
                                ))
                            fig.update_layout(
                                title='Average SOC by Vehicle Category',
                                xaxis_range=[0, simulation_time],
                                xaxis_title='Timestep',
                                yaxis_title='SOC (kWh)'
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"soc_chart_{current_timestep}")

                        with tab3:
                            # Vehicle 0 SOC tracker
                            vehicle0_data = results_df[results_df['vehicle_id'] == 0]
                            chart_data = pd.DataFrame(index=range(simulation_time))
                            chart_data['timestep'] = range(simulation_time)
                            chart_data['soc'] = 0
                            if not vehicle0_data.empty:
                                chart_data.update(vehicle0_data[['timestep', 'soc']].set_index('timestep'))
                            
                            fig = px.line(chart_data, x='timestep', y='soc',
                                         title='Vehicle 0 SOC Over Time')
                            fig.update_layout(
                                xaxis_range=[0, simulation_time],
                                xaxis_title='Timestep',
                                yaxis_title='SOC (kWh)'
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"vehicle0_soc_chart_{current_timestep}")

                        with tab4:
                            # Station 0 inventory and charging slots tracker
                            station0_data = pd.DataFrame(station_history)
                            if not station0_data.empty and 'station_id' in station0_data.columns:
                                station0_data = station0_data[station0_data['station_id'] == 0]
                                chart_data = pd.DataFrame(index=range(simulation_time))
                                chart_data['timestep'] = range(simulation_time)
                                chart_data['inventory'] = 0
                                chart_data['charging_slots_used'] = 0
                                if not station0_data.empty:
                                    chart_data.update(station0_data[['timestep', 'inventory', 'charging_slots_used']].set_index('timestep'))
                                
                                # Create figure with secondary y-axis
                                fig = go.Figure()
                                
                                # Add inventory line
                                fig.add_trace(
                                    go.Scatter(
                                        x=chart_data['timestep'],
                                        y=chart_data['inventory'],
                                        name='Inventory',
                                        line=dict(color='blue')
                                    )
                                )
                                
                                # Add charging slots line
                                fig.add_trace(
                                    go.Scatter(
                                        x=chart_data['timestep'],
                                        y=chart_data['charging_slots_used'],
                                        name='Charging Slots in Use',
                                        line=dict(color='orange')
                                    )
                                )
                                
                                fig.update_layout(
                                    title='Station 0 Status Over Time',
                                    xaxis_title='Timestep',
                                    yaxis_title='Number of Batteries',
                                    xaxis_range=[0, simulation_time],
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"station0_status_chart_{current_timestep}")
                            else:
                                st.warning("No data available for Station 0")

                # Store results at regular intervals
                if t % sample_rate == 0:
                    results.append({
                        'timestep': current_timestep,
                        'hour': current_hour,
                        'vehicle_id': vehicle.vehicle_id,
                        'category': vehicle.category.name,
                        'position': vehicle.position,
                        'soc': vehicle.soc,
                        'needs_swap': vehicle.needs_swap,
                        'out_of_charge': vehicle.out_of_charge,
                        'emergency_swap': vehicle.emergency_swaps > 0
                    })

                # Update progress (ensure it doesn't exceed 1.0)
                progress = min(1.0, (cycle * TIMESTEPS_PER_CYCLE + t + 1) / simulation_time)
                progress_bar.progress(progress)
                time.sleep(0.1)  # Consider removing or reducing this sleep time

                if current_timestep >= simulation_time - 1:
                    break

            # End of cycle - reset vehicles
            for vehicle in vehicles:
                vehicle.reset_for_new_day()
                
            cycle += 1
        
        # Export data
        vehicle_df = pd.DataFrame(results)
        station_df = pd.DataFrame(station_history)
        
        st.subheader("Download Simulation Data")

        # Export data to CSVs
        vehicle_csv = export_vehicle_data(vehicle_df)
        station_csv = export_station_data(station_df)

        # Create zip file containing both CSVs
        def create_zip_file():
            zip_filename = "simulation_data.zip"
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                zipf.write(vehicle_csv)
                zipf.write(station_csv)
            return zip_filename

        # Create zip file and offer for download
        zip_file = create_zip_file()
        with open(zip_file, 'rb') as f:
            st.download_button(
                label="Download All Simulation Data",
                data=f,
                file_name="simulation_data.zip",
                mime="application/zip"
            )

        # Clean up temporary files
        os.remove(vehicle_csv)
        os.remove(station_csv)
        os.remove(zip_file)

# Entry Point
if __name__ == "__main__":
    main()
