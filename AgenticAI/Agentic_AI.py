import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime, timedelta

load_dotenv()
api_key = os.getenv("WEATHER_API_KEY", "YOUR_WEATHERSTACK_KEY")
hotel_api_key= os.getenv("HOTEL_API_KEY", "HOTEL_API_KEY")
mapping_api_key = os.getenv("MAPPING_API_KEY", "MAPPING_API_KEY")


# Step 1 - Models
llm1 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512
)
model1 = ChatHuggingFace(llm=llm1)

search_tool = DuckDuckGoSearchRun()

@tool
def weather(city: str) -> str:
    """Return current weather of a city using Weatherstack API."""
    try:
        url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
        response = requests.get(url)
        data = response.json()
        if 'current' in data:
            temp = data['current']['temperature']
            desc = data['current']['weather_descriptions'][0]
            return f"Weather in {city}: {temp}°C, {desc}"
        return str(data)
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

def get_city_id(city_name: str):
    """Get city ID from city name using mapping API."""
    try:
        url = "https://api.makcorps.com/mapping"
        params = {
            'api_key': mapping_api_key,
            'name': city_name
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            json_data = response.json()
            if json_data and len(json_data) > 0:
                # Return the first matching city ID
                return json_data[0].get("document_id")
        return None
    except Exception as e:
        st.error(f"Error getting city ID: {str(e)}")
        return None

def get_hotels(city_id: str, checkin: str, checkout: str, rooms: int, adults: int):
    """Return hotels in a city using Makcorps API."""
    try:
        url = "https://api.makcorps.com/city"
        params = {
            'cityid': city_id,
            'pagination': '0',
            'cur': 'USD',
            'rooms': str(rooms),
            'adults': str(adults),
            'checkin': checkin,
            'checkout': checkout,
            'api_key': hotel_api_key
        }
        response = requests.get(url, params=params)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Custom CSS for better styling
st.markdown("""
    <style>
    .hotel-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .hotel-name {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .hotel-price {
        font-size: 28px;
        font-weight: bold;
        color: #ffd700;
    }
    .hotel-rating {
        font-size: 18px;
        color: #ffd700;
    }
    .hotel-info {
        font-size: 14px;
        margin: 5px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 25px;
        border: none;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 AI Travel Information Provider")
st.markdown("---")

user_choice = st.selectbox(
    "What would you like to explore?",
    ['🔍 Search Places to Travel', '🌤️ Look up Weather', '🏨 Find Hotels'],
    index=0
)

# 1 - Search Places
if user_choice == "🔍 Search Places to Travel":
    st.subheader("🗺️ Discover Travel Destinations")
    query = st.text_input("Enter a place or travel destination:", placeholder="e.g., Paris, Bali, Tokyo")
    
    if st.button("🔎 Search"):
        with st.spinner("Searching for amazing places..."):
            try:
                result = search_tool.run(query)
                st.success("✅ Search Results:")
                st.info(result)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# 2 - Weather
elif user_choice == "🌤️ Look up Weather":
    st.subheader("☀️ Check Weather Conditions")
    city_name = st.text_input("Enter city name:", placeholder="e.g., New York, London, Dubai")
    
    if st.button("🌡️ Get Weather"):
        with st.spinner("Fetching weather data..."):
            try:
                result = weather.invoke(city_name)
                st.success("✅ Weather Information:")
                st.info(result)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# 3 - Hotels
elif user_choice == "🏨 Find Hotels":
    st.subheader("🏨 Search Hotels")
    
    col1, col2 = st.columns(2)
    
    with col1:
        city_name = st.text_input("City:", placeholder="e.g., Arlington, London, Paris")
        tomorrow = datetime.now() + timedelta(days=1)
        checkin_date = st.date_input("Check-in:", value=tomorrow, min_value=datetime.now().date())
        rooms = st.number_input("Rooms:", min_value=1, value=1)
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        day_after = datetime.now() + timedelta(days=2)
        checkout_date = st.date_input("Check-out:", value=day_after, min_value=datetime.now().date())
        adults = st.number_input("Adults:", min_value=1, value=2)
    
    if st.button("🔍 Search Hotels"):
        if checkout_date <= checkin_date:
            st.error("❌ Check-out date must be after check-in date!")
        elif not city_name.strip():
            st.error("❌ Please enter a city name!")
        else:
            with st.spinner(f"Finding city ID for {city_name}..."):
                city_id = get_city_id(city_name)
                
                if not city_id:
                    st.error(f"❌ Could not find city: {city_name}. Please try another city name.")
                else:
                    st.info(f"✅ Found city! (ID: {city_id})")
                    
                    with st.spinner("Searching for the best hotels..."):
                        try:
                            checkin_str = checkin_date.strftime('%Y-%m-%d')
                            checkout_str = checkout_date.strftime('%Y-%m-%d')
                            
                            result = get_hotels(city_id, checkin_str, checkout_str, int(rooms), int(adults))
                            
                            if isinstance(result, list) and len(result) > 0:
                                st.success(f"✅ Found {len(result)} hotels in {city_name}!")
                                st.markdown("---")
                                
                                # Display hotels in a nice format
                                for idx, hotel in enumerate(result[:10], 1):  # Show top 10
                                    with st.container():
                                        col1, col2, col3 = st.columns([3, 1, 1])
                                        
                                        with col1:
                                            st.markdown(f"### 🏨 {hotel.get('name', 'N/A')}")
                                            rating = hotel.get('reviews', {}).get('rating', 0)
                                            review_count = hotel.get('reviews', {}).get('count', 0)
                                            st.markdown(f"⭐ **{rating}/5** ({review_count:,} reviews)")
                                            
                                            if 'telephone' in hotel:
                                                st.markdown(f"📞 {hotel['telephone']}")
                                        
                                        with col2:
                                            st.markdown(f"### 💰 ${hotel.get('price1', 'N/A')}")
                                            st.caption("per night")
                                        
                                        with col3:
                                            vendor = hotel.get('vendor1', 'Book Now')
                                            st.markdown(f"**{vendor}**")
                                            if rating >= 4.5:
                                                st.success("🌟 Top Rated")
                                            elif rating >= 4.0:
                                                st.info("👍 Great")
                                        
                                        # Location info
                                        if 'geocode' in hotel:
                                            lat = hotel['geocode'].get('latitude', 'N/A')
                                            lon = hotel['geocode'].get('longitude', 'N/A')
                                            st.caption(f"📍 Location: {lat}, {lon}")
                                        
                                        st.markdown("---")
                            
                            elif isinstance(result, dict) and not result.get('success', True):
                                st.warning(f"⚠️ {result.get('message', 'No hotels found')}")
                            else:
                                st.warning(f"⚠️ No hotels found in {city_name} for your search criteria.")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("Powered by AI Travel Assistant 🤖")