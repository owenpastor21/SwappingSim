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

# Configuration Constants
max_history_length = 1000  # Reduced from 5000
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
    'Parañaque': (14.4793, 121.0198),
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

# Define Vehicle Categories
class VehicleCategory:
    """Defines different types of electric vehicles with specific characteristics
    
    Attributes:
        name: Category identifier (e.g., 'Type A', 'Type B')
        battery_capacity: Maximum battery capacity in kWh
        daily_range: Expected daily travel distance in km
        dod_threshold: Depth of Discharge threshold as percentage before requiring swap
        consumption_rate: Energy consumption in kWh per km
    """
    def __init__(self, name, battery_capacity, daily_range, dod_threshold, consumption_rate):
        self.name = name
        self.battery_capacity = battery_capacity  # in kWh
        self.daily_range = daily_range  # in km
        self.dod_threshold = dod_threshold  # Depth of Discharge threshold (%)
        self.consumption_rate = consumption_rate  # kWh per km

# Define Station Categories
class StationCategory:
    """Defines different types of battery swapping stations
    
    Attributes:
        name: Station type identifier (e.g., 'Small', 'Medium', 'Large')
        charging_speed: Rate of battery charging in kW
        inventory_capacity: Maximum number of batteries that can be stored
        charging_slots: Number of simultaneous charging positions
        battery_types: List of vehicle categories this station can service
    """
    def __init__(self, name, charging_speed, inventory_capacity, charging_slots, battery_types=None):
        self.name = name
        self.charging_speed = charging_speed  # in kW
        self.inventory_capacity = inventory_capacity  # number of batteries
        self.charging_slots = charging_slots  # number of slots
        self.battery_types = battery_types or []  # list of compatible vehicle categories

# Vehicle Class
class Vehicle:
    """Represents an individual electric vehicle in the simulation
    
    Handles vehicle movement, battery consumption, and swap requests.
    Tracks position, state of charge, and route information.
    """
    def __init__(self, vehicle_id, category, initial_soc, route):
        self.vehicle_id = vehicle_id
        self.category = category
        self.soc = initial_soc  # State of Charge in kWh
        self.route = route  # List of node IDs
        self.route_index = 0
        self.position = route[0]
        self.needs_swap = False
        self.distance_traveled = 0  # in km
        self.out_of_charge = False
        self.out_of_charge_duration = 0  # Track how long vehicle has been depleted
        self.emergency_swaps = 0  # Track number of emergency swaps
        self.last_path_update = 0  # Track when path was last updated
        self.route_lock = threading.Lock()  # Lock for thread-safe route updates

    def check_emergency_swap(self, G, stations):
        """Check if vehicle is eligible for emergency swap and perform it if possible"""
        if self.out_of_charge:
            self.out_of_charge_duration += 1
            if self.out_of_charge_duration >= TIMESTEPS_FOR_EMERGENCY:
                # Find nearest station for emergency swap
                nearest_station = find_nearest_station(G, self.position, stations)
                if nearest_station and nearest_station.accept_battery(self):
                    self.soc = self.category.battery_capacity  # Full charge
                    self.out_of_charge = False
                    self.needs_swap = False
                    self.out_of_charge_duration = 0
                    self.emergency_swaps += 1
                    return True
        return False

    def move(self, G, stations):
        """Updates vehicle position and energy consumption based on movement
        
        Args:
            G: NetworkX graph representing the road network
            stations: List of all Station objects
        
        Note:
            Energy consumption is calculated based on the distance traveled 
            during the time step (5 minutes)
        """
        # First check for emergency swap
        if self.check_emergency_swap(G, stations):
            return

        if self.out_of_charge:
            return

        if self.route_index < len(self.route) - 1:
            prev_position = self.position
            self.route_index += 1
            self.position = self.route[self.route_index]

            edge_data = G.get_edge_data(prev_position, self.position)
            if edge_data:
                distance = edge_data[0]['length'] / 1000  # Convert meters to km
                
                # Calculate actual distance covered in this time step
                # Assuming average speed of 30 km/h in urban areas
                average_speed = 30  # km/h
                time_fraction = TIME_STEP_MINUTES / MINUTES_PER_HOUR  # convert minutes to hours
                actual_distance = min(distance, average_speed * time_fraction)
                
                self.distance_traveled += actual_distance
                energy_consumed = actual_distance * self.category.consumption_rate
                self.soc -= energy_consumed

                if self.soc <= self.category.battery_capacity * self.category.dod_threshold:
                    self.needs_swap = True
                if self.soc <= 0:
                    self.out_of_charge = True
                    self.soc = 0
        else:
            self.route = generate_random_route(G)
            self.route_index = 0

    def swap_battery(self, station):
        if station.accept_battery(self):
            self.soc = self.category.battery_capacity  # Set SOC to 100%
            self.needs_swap = False
            return True
        return False

    def get_soc_color(self):
        # Return a color based on SOC percentage
        soc_percentage = (self.soc / self.category.battery_capacity) * 100
        if soc_percentage > 75:
            return 'green'
        elif soc_percentage > 50:
            return 'yellowgreen'
        elif soc_percentage > 25:
            return 'orange'
        else:
            return 'red'

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
            nearest_station = find_nearest_station(G, self.position, stations)
            if nearest_station:
                try:
                    return nx.shortest_path(G, self.position, nearest_station.location, weight='length')
                except nx.NetworkXNoPath:
                    pass
        return generate_random_route(G)

# Station Class
class Station:
    """Represents a battery swapping station
    
    Manages battery inventory, charging operations, and swap requests.
    """
    def __init__(self, station_id, category, location):
        self.station_id = station_id
        self.category = category
        self.location = location  # Node ID on the network
        self.inventory = category.inventory_capacity  # Current battery inventory
        self.charging_slots = category.charging_slots  # Number of charging slots
        self.batteries_charging = 0  # Number of batteries currently charging

    def accept_battery(self, vehicle):
        if vehicle.category.name not in self.category.battery_types:
            return False  # Station can't handle this type of battery
        if self.inventory > 0:
            self.inventory -= 1
            self.batteries_charging += 1
            return True
        return False

    def charge_batteries(self):
        # Simple logic to simulate charging batteries
        if self.batteries_charging > 0:
            self.batteries_charging -= 1
            self.inventory += 1

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
    """Locates the closest available battery swapping station
    
    Args:
        G: NetworkX graph representing the road network
        vehicle_position: Current node ID of vehicle
        stations: List of all Station objects
    Returns:
        Station object of nearest station or None if unreachable
    """
    min_distance = float('inf')
    nearest_station = None
    for station in stations:
        try:
            distance = nx.shortest_path_length(G, vehicle_position, station.location, weight='length')
            if distance < min_distance:
                min_distance = distance
                nearest_station = station
        except nx.NetworkXNoPath:
            continue
    return nearest_station

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

    # Add vehicles with ID tooltips
    for vehicle in vehicles:
        lon, lat = G.nodes[vehicle.position]['x'], G.nodes[vehicle.position]['y']
        popup_text = f"""
        Vehicle ID: {vehicle.vehicle_id}
        Category: {vehicle.category.name}
        SOC: {vehicle.soc:.2f} kWh
        Needs Swap: {'Yes' if vehicle.needs_swap else 'No'}
        Out of Charge: {'Yes' if vehicle.out_of_charge else 'No'}
        """
        folium.CircleMarker(
            location=(lat, lon),
            radius=3,
            color=vehicle.get_soc_color(),
            fill=True,
            fill_opacity=0.9,
            popup=popup_text,
            tooltip=f"V{vehicle.vehicle_id}"  # Added vehicle ID tooltip
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

    # Add city selection and radius control to sidebar
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
        type_a_capacity = st.slider("Battery Capacity (kWh)", 20, 80, 50, key="type_a_cap")
        type_a_range = st.slider("Daily Range (km)", 100, 300, 200, key="type_a_range")
        type_a_dod = st.slider("Depth of Discharge Threshold", 0.1, 0.6, 0.4, key="type_a_dod")
        type_a_consumption = st.slider("Consumption Rate (kWh/km)", 1.0, 10.0, 5.15, key="type_a_cons")
        vehicle_categories['Type A'] = VehicleCategory('Type A', type_a_capacity, type_a_range, type_a_dod, type_a_consumption)
        
        # Type B Config
        st.subheader("Type B (Medium Vehicles)")
        type_b_capacity = st.slider("Battery Capacity (kWh)", 40, 100, 75, key="type_b_cap")
        type_b_range = st.slider("Daily Range (km)", 200, 400, 300, key="type_b_range")
        type_b_dod = st.slider("Depth of Discharge Threshold", 0.1, 0.6, 0.4, key="type_b_dod")
        type_b_consumption = st.slider("Consumption Rate (kWh/km)", 5.0, 20.0, 15.25, key="type_b_cons")
        vehicle_categories['Type B'] = VehicleCategory('Type B', type_b_capacity, type_b_range, type_b_dod, type_b_consumption)
        
        # Type C Config
        st.subheader("Type C (Heavy Vehicles)")
        type_c_capacity = st.slider("Battery Capacity (kWh)", 60, 150, 100, key="type_c_cap")
        type_c_range = st.slider("Daily Range (km)", 300, 500, 400, key="type_c_range")
        type_c_dod = st.slider("Depth of Discharge Threshold", 0.1, 0.6, 0.4, key="type_c_dod")
        type_c_consumption = st.slider("Consumption Rate (kWh/km)", 10.0, 30.0, 25.35, key="type_c_cons")
        vehicle_categories['Type C'] = VehicleCategory('Type C', type_c_capacity, type_c_range, type_c_dod, type_c_consumption)

    # Station Categories Configuration
    with st.sidebar.expander("Station Categories Configuration"):
        station_categories = {}
        
        # Small Station Config
        st.subheader("Small Stations")
        small_charging = st.slider("Charging Speed (kW)", 20, 80, 50, key="small_charging")
        small_inventory = st.slider("Inventory Capacity", 10, 30, 20, key="small_inventory")
        small_slots = st.slider("Charging Slots", 2, 8, 5, key="small_slots")
        station_categories['Small'] = StationCategory('Small', small_charging, small_inventory, small_slots, ['Type A'])
        
        # Medium Station Config
        st.subheader("Medium Stations")
        medium_charging = st.slider("Charging Speed (kW)", 50, 150, 100, key="medium_charging")
        medium_inventory = st.slider("Inventory Capacity", 20, 60, 45, key="medium_inventory")
        medium_slots = st.slider("Charging Slots", 5, 15, 10, key="medium_slots")
        station_categories['Medium'] = StationCategory('Medium', medium_charging, medium_inventory, medium_slots, ['Type A', 'Type B'])
        
        # Large Station Config
        st.subheader("Large Stations")
        large_charging = st.slider("Charging Speed (kW)", 100, 200, 150, key="large_charging")
        large_inventory = st.slider("Inventory Capacity", 40, 120, 90, key="large_inventory")
        large_slots = st.slider("Charging Slots", 10, 30, 20, key="large_slots")
        station_categories['Large'] = StationCategory('Large', large_charging, large_inventory, large_slots, ['Type A', 'Type B', 'Type C'])

    # Rest of sidebar inputs
    st.sidebar.header("Simulation Parameters")
    num_vehicles = st.sidebar.number_input("Number of Vehicles", min_value=1, value=300)
    
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
        num_stations = st.sidebar.number_input("Number of Swapping Stations", min_value=1, value=30)
        
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
                
                # Display area info
                st.info(f"Simulating area centered on {center_city} with {radius_meters}m radius")
                
                # Create initial map with dynamic zoom level
                center_lat, center_lon = NCR_CITIES[center_city]
                zoom_level = get_zoom_level(radius_meters)
                initial_map = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=zoom_level
                )
                
                # Add circle to show radius
                folium.Circle(
                    location=[center_lat, center_lon],
                    radius=radius_meters,
                    color='red',
                    fill=True,
                    fillOpacity=0.1
                ).add_to(initial_map)
                
                folium_static(initial_map)
                
                # Initialize vehicles within bounds
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
                            route=route
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
        for t in range(simulation_time):
            # Update paths in parallel
            update_vehicle_paths(vehicles, G, stations, t)
            
            # Move vehicles and handle battery swaps
            for vehicle in vehicles:
                vehicle.move(G, stations)
                if vehicle.needs_swap and not vehicle.out_of_charge:
                    nearest_station = find_nearest_station(G, vehicle.position, stations)
                    if nearest_station and vehicle.position == nearest_station.location:
                        if vehicle.swap_battery(nearest_station):
                            # Update path after successful battery swap
                            new_route = vehicle.calculate_new_path(G, None)  # Generate random route after swap
                            vehicle.update_route(G, new_route)
                            vehicle.last_path_update = t

            # Update station charging slots
            for station in stations:
                station.charge_batteries()

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
            
            # Only update visualizations periodically
            if t % VISUALIZATION_UPDATE_INTERVAL == 0:
                # Update map
                simulation_map = create_map(G, vehicles, stations)
                with map_placeholder.container():
                    folium_static(simulation_map)

                # Update metrics and charts
                with metrics_placeholder.container():
                    cols = st.columns(len(selected_vehicle_categories) + 2)  # +2 for out of charge and emergency swaps
                    cols[0].metric("Vehicles Out of Charge", num_out_of_charge)
                    cols[1].metric("Emergency Swaps", total_emergency_swaps)
                    for i, category in enumerate(selected_vehicle_categories, 2):
                        cols[i].metric(f"{category} Avg SOC", f"{category_soc.get(category, 0):.2f} kWh")

                # Update charts
                if len(results_df) > 0:
                    with chart_placeholder.container():
                        tab1, tab2, tab3 = st.tabs(["Vehicles Out of Charge", "Average SOC by Category", "Emergency Swaps"])
                        with tab1:
                            st.line_chart(data=results_df, x='Time', y='Vehicles Out of Charge')
                        with tab2:
                            soc_columns = [f'Avg SOC {cat}' for cat in selected_vehicle_categories]
                            st.line_chart(data=results_df, x='Time', y=soc_columns)
                        with tab3:
                            st.line_chart(data=results_df, x='Time', y='Total Emergency Swaps')

            # Store results at regular intervals
            if t % sample_rate == 0:
                results.append({
                    'Time': t,
                    'Vehicles Out of Charge': num_out_of_charge,
                    'Total Emergency Swaps': total_emergency_swaps,
                    **{f'Avg SOC {cat}': category_soc.get(cat, 0) for cat in selected_vehicle_categories}
                })

            # Update progress
            progress_bar.progress((t + 1) / simulation_time)
            time.sleep(0.1)  # Consider removing or reducing this sleep time

        # For final results, create a fresh DataFrame with all data points
        st.subheader("Simulation Results")
        final_results_df = pd.DataFrame(results)
        final_results_df['Time'] = range(simulation_time - len(results), simulation_time)
        st.line_chart(final_results_df.set_index('Time'))

        # Display Final Map
        st.subheader("Final State Map")
        simulation_map = create_map(G, vehicles, stations)
        folium_static(simulation_map)

# Entry Point
if __name__ == "__main__":
    main()